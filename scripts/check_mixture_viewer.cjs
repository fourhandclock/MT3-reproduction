// Optional local-browser verification. Set PLAYWRIGHT_MODULE/BROWSER_EXECUTABLE when needed.
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL, fileURLToPath} = require('node:url');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

const workspace = path.resolve(__dirname, '..');
const output = path.join(workspace, 'outputs', 'mixtures');
const downloadsPath = path.join(workspace, 'tmp', 'browser-downloads');
const expectedConditions = [
  'clean', 'separated_clean', 'accompaniment_only',
  'target_+6', 'mixture_+6', 'separated_+6',
  'target_0', 'mixture_0', 'separated_0',
  'target_-6', 'mixture_-6', 'separated_-6',
];
const validation = {
  passed: false, clips: 0, conditions_per_clip: expectedConditions.length,
  combinations: 0, audio_metadata_checks: 0, export_link_checks: 0,
  accompaniment_only_checks: 0, accompaniment_false_positive_notes: 0,
  comparison_checks: [], javascript_errors: [],
};

function savedExport(url) {
  const filename = fileURLToPath(url);
  const relative = path.relative(output, filename);
  assert(!relative.startsWith('..') && !path.isAbsolute(relative), 'Export link escapes this experiment.');
  assert(fs.existsSync(filename), 'Missing linked export: ' + relative);
  assert(fs.statSync(filename).size > 0, 'Empty linked export: ' + relative);
  return filename;
}

async function loadAudioMetadata(page) {
  // Request metadata through the same local-file path that playback uses,
  // without playing 144 recordings during validation.
  const expectedSource = await page.locator('#audio').evaluate(audio => {
    audio.load();
    return audio.src;
  });
  await page.waitForFunction(source => {
    const audio = document.getElementById('audio');
    return !audio.error && audio.currentSrc === source && audio.readyState >= 1 &&
      Number.isFinite(audio.duration) && audio.duration > 0;
  }, expectedSource, {timeout: 20000});
  const duration = await page.locator('#audio').evaluate(audio => audio.duration);
  assert(Math.abs(duration - 12) < 0.02, 'Expected twelve-second local audio, received ' + duration);
  return duration;
}

async function waitForTime(page, seconds) {
  await page.waitForFunction(target => {
    const audio = document.getElementById('audio');
    return !audio.seeking && Math.abs(audio.currentTime - target) < 0.05;
  }, seconds);
  await page.waitForFunction(target => {
    const line = document.getElementById('playhead');
    // The scored roll is 10 seconds, x=45..990 in its 1000-wide viewBox.
    return line && Math.abs(Number(line.getAttribute('x1')) - (45 + target / 10 * 945)) < 5;
  }, seconds);
}

async function seek(page, seconds) {
  await page.locator('#audio').evaluate((audio, target) => { audio.currentTime = target; }, seconds);
  await waitForTime(page, seconds);
}

async function compare(page, source, button, expected, seconds) {
  await page.selectOption('#condition', source);
  await loadAudioMetadata(page);
  await seek(page, seconds);
  await page.click(button);
  assert.equal(await page.locator('#condition').inputValue(), expected, 'Wrong comparison condition.');
  // Do not call audio.load() here: its extra reset would mask time-preservation
  // bugs in the viewer's own onloadedmetadata handler.
  const expectedSource = await page.locator('#audio').evaluate(audio => audio.src);
  await page.waitForFunction(sourceUrl => {
    const audio = document.getElementById('audio');
    return audio.currentSrc === sourceUrl && audio.readyState >= 1 && Number.isFinite(audio.duration);
  }, expectedSource, {timeout: 20000});
  await waitForTime(page, seconds);
  validation.comparison_checks.push({source, target: expected, seek_seconds: seconds,
    condition_correct: true, playback_time_preserved: true, playhead_updated: true});
}

(async () => {
  fs.mkdirSync(downloadsPath, {recursive: true});
  fs.mkdirSync(path.join(output, 'example'), {recursive: true});
  const browser = await chromium.launch({
    headless: true, downloadsPath,
    ...(process.env.BROWSER_EXECUTABLE ? {executablePath: process.env.BROWSER_EXECUTABLE} : {}),
  });
  let stage = 'open viewer';
  try {
    const page = await browser.newPage({viewport: {width: 1280, height: 1050}, acceptDownloads: true});
    page.setDefaultTimeout(15000);
    page.on('pageerror', error => validation.javascript_errors.push(String(error)));
    await page.goto(pathToFileURL(path.join(output, 'review.html')).href);
    const clips = await page.locator('#clip option').evaluateAll(options => options.map(option => option.value));
    assert.equal(clips.length, 12, 'Expected twelve excerpt selectors.');
    assert.equal(new Set(clips).size, 12, 'Duplicate excerpt selector.');
    validation.clips = clips.length;
    const durations = [];
    stage = 'verify every excerpt and condition';
    for (const clip of clips) {
      await page.selectOption('#clip', clip);
      const conditions = await page.locator('#condition option').evaluateAll(options => options.map(option => option.value));
      assert.deepEqual([...conditions].sort(), [...expectedConditions].sort(), 'Wrong condition inventory for ' + clip);
      for (const condition of expectedConditions) {
        await page.selectOption('#condition', condition);
        durations.push(await loadAudioMetadata(page));
        validation.audio_metadata_checks++;
        assert.equal(await page.locator('#roll svg').count(), 1, 'Missing note plot: ' + clip + '/' + condition);
        assert.equal(await page.locator('#playhead').count(), 1, 'Missing playhead: ' + clip + '/' + condition);
        const aria = await page.locator('#roll svg').getAttribute('aria-label');
        assert(aria.includes(clip) && aria.includes(condition), 'Plot did not update with selectors.');
        const metrics = await page.locator('#metrics').innerText();
        const noGuitar = condition === 'accompaniment_only';
        assert(metrics.includes(noGuitar ? 'false-positive notes' : 'Onset F1'), 'Missing condition metrics.');
        assert.equal(await page.locator('#compare').isDisabled(), noGuitar || condition === 'clean' || condition === 'separated_clean');
        assert.equal(await page.locator('#separate').isDisabled(), noGuitar || condition === 'clean' || condition === 'separated_clean');
        if (noGuitar) {
          const match = metrics.match(/^(\d+) false-positive notes/);
          assert(match, 'No-guitar control lacks a false-positive count.');
          validation.accompaniment_false_positive_notes += Number(match[1]);
          validation.accompaniment_only_checks++;
        }
        for (const link of ['json', 'midi', 'csv']) {
          const url = await page.locator('#' + link).evaluate(anchor => anchor.href);
          const filename = savedExport(url);
          assert.equal(path.basename(filename, path.extname(filename)), clip, 'Export link is from another excerpt.');
          if (link === 'json') {
            const document = JSON.parse(fs.readFileSync(filename, 'utf8'));
            assert(Array.isArray(document.notes), 'Note JSON schema missing notes array.');
            assert.equal(document.metadata.condition, condition, 'Note JSON is from another condition.');
            assert.equal(document.time_unit, 'seconds', 'Unexpected raw note time units.');
          }
          validation.export_link_checks++;
        }
        assert.equal(await page.locator('#status').innerText(), '', 'Viewer reports an audio error.');
        validation.combinations++;
      }
      console.log('Viewer verified:', clip, '(12 conditions and local audio files).');
    }
    assert.equal(validation.combinations, 144);
    assert.equal(validation.accompaniment_only_checks, 12);
    validation.audio_duration_min_seconds = Math.min(...durations);
    validation.audio_duration_max_seconds = Math.max(...durations);
    validation.metrics_and_svg_all_combinations = true;
    stage = 'click plot and verify playhead';
    await page.selectOption('#clip', clips[0]);
    await page.selectOption('#condition', 'mixture_0');
    await loadAudioMetadata(page);
    const plot = await page.locator('#roll svg').boundingBox();
    assert(plot, 'Note plot has no visible dimensions.');
    const targetSeconds = 3;
    const viewBoxX = 45 + targetSeconds / 10 * 945;
    await page.locator('#roll svg').click({position: {x: plot.width * viewBoxX / 1000, y: plot.height / 2}});
    await waitForTime(page, targetSeconds);
    validation.click_seek_and_playhead = true;
    stage = 'paired comparison buttons and preserved position';
    for (const suffix of ['+6', '0', '-6']) {
      await compare(page, 'target_' + suffix, '#compare', 'mixture_' + suffix, 2.75);
      await compare(page, 'mixture_' + suffix, '#compare', 'target_' + suffix, 2.75);
      await compare(page, 'mixture_' + suffix, '#separate', 'separated_' + suffix, 4.25);
      await compare(page, 'separated_' + suffix, '#separate', 'mixture_' + suffix, 4.25);
    }
    validation.comparison_button_checks = validation.comparison_checks.length;
    stage = 'download note JSON';
    await page.selectOption('#condition', 'separated_0');
    await loadAudioMetadata(page);
    const downloadEvent = page.waitForEvent('download');
    await page.click('#json');
    const download = await downloadEvent;
    const downloadPath = path.join(output, 'example', 'viewer-notes.json');
    await download.saveAs(downloadPath);
    const document = JSON.parse(fs.readFileSync(downloadPath, 'utf8'));
    assert(Array.isArray(document.notes) && document.notes.length > 0, 'Downloaded notes missing.');
    assert.equal(document.metadata.condition, 'separated_0', 'Downloaded wrong condition.');
    const originalUrl = await page.locator('#json').evaluate(anchor => anchor.href);
    assert.deepEqual(document, JSON.parse(fs.readFileSync(savedExport(originalUrl), 'utf8')),
      'Browser download metadata/events differ from the exported note JSON.');
    validation.note_json_download = true;
    validation.downloaded_note_count = document.notes.length;
    validation.downloaded_clip = clips[0];
    validation.downloaded_condition = 'separated_0';
    stage = 'screenshot and final verification';
    await seek(page, 3);
    await page.screenshot({path: path.join(output, 'review.png'), fullPage: true});
    validation.screenshot = 'review.png';
    assert.deepEqual(validation.javascript_errors, [], 'Viewer emitted JavaScript errors.');
    validation.passed = true;
    console.log('Viewer passed: 144 combinations, local audio, plot seek, 12 comparison checks, and note download.');
  } catch (error) {
    validation.failed_stage = stage;
    validation.failure = String(error && error.stack || error);
    throw error;
  } finally {
    validation.completed_at = new Date().toISOString();
    fs.writeFileSync(path.join(output, 'viewer-validation.json'), JSON.stringify(validation, null, 2));
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
