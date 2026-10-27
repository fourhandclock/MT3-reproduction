// Optional local-browser check. Set PLAYWRIGHT_MODULE/BROWSER_EXECUTABLE as needed.
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL, fileURLToPath} = require('node:url');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

const workspace = path.resolve(__dirname, '..');
const output = path.join(workspace, 'outputs', 'streaming');
const downloadsPath = path.join(workspace, 'tmp', 'browser-downloads', 'streaming');
const expectedModes = ['causal_20ms', 'causal_40ms', 'causal_80ms', 'offline_bidir'];
const validation = {
  passed: false, clips: 0, modes_per_clip: 4, combinations: 0,
  audio_metadata_checks: 0, export_link_checks: 0, note_json_download_checks: 0,
  source_delivery_chart_checks: 0, virtual_delivery_event_checks: 0,
  causal_event_log_checks: 0, eof_flush_tags_checked: 0,
  comparison_checks: [], javascript_errors: [],
};

function savedExport(url) {
  const filename = fileURLToPath(url);
  const relative = path.relative(output, filename);
  assert(!relative.startsWith('..') && !path.isAbsolute(relative), 'Export escapes this experiment.');
  assert(fs.existsSync(filename), 'Missing linked export: ' + relative);
  assert(fs.statSync(filename).size > 0, 'Empty linked export: ' + relative);
  return filename;
}

async function loadAudioMetadata(page, clip) {
  const source = await page.locator('#audio').evaluate(audio => {
    audio.load();
    return audio.src;
  });
  await page.waitForFunction(expected => {
    const audio = document.getElementById('audio');
    return !audio.error && audio.currentSrc === expected && audio.readyState >= 1 &&
      Number.isFinite(audio.duration) && audio.duration > 0;
  }, source, {timeout: 20000});
  assert.equal(path.basename(fileURLToPath(source)), clip + '.wav', 'Wrong source audio excerpt.');
  const duration = await page.locator('#audio').evaluate(audio => audio.duration);
  assert(Math.abs(duration - 12) < 0.02, 'Expected twelve-second local PCM, received ' + duration);
  return duration;
}

async function waitForTime(page, seconds) {
  await page.waitForFunction(target => {
    const audio = document.getElementById('audio');
    return !audio.seeking && Math.abs(audio.currentTime - target) < 0.05;
  }, seconds);
  await page.waitForFunction(target => {
    const source = document.getElementById('source-playhead');
    const delivery = document.getElementById('delivery-playhead');
    const svg = document.querySelector('#events svg');
    const duration = Number(svg && svg.dataset.duration) || 12;
    return source && delivery &&
      Math.abs(Number(source.getAttribute('x1')) - (45 + Math.min(target, 10) / 10 * 945)) < 5 &&
      Math.abs(Number(delivery.getAttribute('x1')) - (60 + Math.min(target, duration) / duration * 925)) < 5;
  }, seconds);
}

async function seek(page, seconds) {
  await page.locator('#audio').evaluate((audio, target) => { audio.currentTime = target; }, seconds);
  await waitForTime(page, seconds);
}

async function inspectEvents(page, record, mode) {
  const view = await page.evaluate(() => ({
    events: current.events,
    duration: Number(document.querySelector('#events svg').dataset.duration),
    marks: [...document.querySelectorAll('#events svg g')].map(group => ({
      title: group.querySelector('title')?.textContent,
      circles: [...group.querySelectorAll('circle')].map(circle => Number(circle.getAttribute('cx'))),
      paths: [...group.querySelectorAll('path')].map(mark => mark.getAttribute('d')),
    })),
  }));
  const sourceEvents = record.events || [];
  assert.equal(view.events.length, sourceEvents.length, 'Viewer event count differs from saved replay.');
  if (mode === 'offline_bidir') {
    assert.equal(view.events.length, 0, 'Offline reference must not invent incremental events.');
    assert((await page.locator('#events svg').textContent()).includes('No incremental delivery events'),
      'Offline delivery chart lacks its explanation.');
    await page.locator('#showall').check();
    assert.equal(await page.locator('#eventrows tr').count(), 0);
    assert((await page.locator('#eventcount').innerText()).includes('no incremental delivery events'));
    return;
  }
  assert(sourceEvents.length > 0, 'Causal result contains no reviewable events: ' + record.id + '/' + mode);
  assert.equal(view.marks.length, sourceEvents.length, 'Not every replay event appears in the timeline.');
  for (let index = 0; index < sourceEvents.length; index++) {
    const raw = sourceEvents[index], shown = view.events[index], mark = view.marks[index];
    assert(Number.isFinite(raw.delivered_at_seconds), 'Missing virtual processing clock timestamp.');
    assert(raw.delivered_at_seconds >= raw.emitted_at_audio_seconds - 1e-9,
      'Virtual completion precedes the packet arrival.');
    assert.equal(shown.delivery, raw.delivered_at_seconds,
      'Delivery chart used packet time instead of measured virtual completion.');
    assert.equal(shown.source, raw.source_time, 'Delivery chart changed the source timestamp.');
    assert.equal(shown.pitch, raw.pitch);
    assert.equal(shown.string, raw.string);
    assert.equal(shown.flush, Boolean(raw.flush));
    assert(mark.title.includes('source ' + raw.source_time.toFixed(3) + 's'));
    assert(mark.title.includes('delivered ' + raw.delivered_at_seconds.toFixed(3) + 's'));
    const sourceX = 60 + raw.source_time / view.duration * 925;
    const deliveredX = 60 + raw.delivered_at_seconds / view.duration * 925;
    assert(Math.abs(mark.circles[0] - sourceX) < 0.015, 'Source marker is at the wrong coordinate.');
    if (raw.type === 'note_on') {
      assert(Math.abs(mark.circles[1] - deliveredX) < 0.015, 'Onset delivery marker uses the wrong clock.');
    } else {
      const diamond = mark.paths.find(item => /l4,4/.test(item));
      assert(diamond, 'Release delivery diamond is missing.');
      const actualX = Number(diamond.match(/^M([\d.\-]+)/)[1]);
      assert(Math.abs(actualX - deliveredX) < 0.015, 'Release delivery marker uses the wrong clock.');
    }
    validation.virtual_delivery_event_checks++;
  }
  await page.locator('#showall').check();
  assert.equal(await page.locator('#eventrows tr').count(), Math.min(40, sourceEvents.length),
    'Final event log does not display the latest replay events.');
  const flush = sourceEvents.filter(event => event.flush);
  if (flush.length) {
    assert((await page.locator('#eventrows').innerText()).includes('(flush)'), 'EOF release lacks its flush tag.');
    validation.eof_flush_tags_checked += flush.length;
  }
  await page.locator('#showall').uncheck();
  await seek(page, 5);
  const delivered = sourceEvents.filter(event => event.delivered_at_seconds <= 5).length;
  await page.waitForFunction(count => document.querySelectorAll('#eventrows tr').length === Math.min(40, count), delivered);
  assert((await page.locator('#eventcount').innerText()).startsWith(delivered + ' of ' + sourceEvents.length + ' events'),
    'Event log is not gated by the virtual delivery clock.');
  validation.causal_event_log_checks++;
}

async function compare(page, source, target, seconds, button = false) {
  await page.selectOption('#mode', source);
  const clip = await page.locator('#clip').inputValue();
  await loadAudioMetadata(page, clip);
  await seek(page, seconds);
  if (button) await page.click('#compare');
  else await page.selectOption('#mode', target);
  assert.equal(await page.locator('#mode').inputValue(), target, 'Wrong comparison mode.');
  // Do not reset/load the audio here: that would mask a viewer time-preservation bug.
  await waitForTime(page, seconds);
  validation.comparison_checks.push({source, target, seconds, button,
    playback_time_preserved: true, source_and_delivery_playheads_updated: true});
}

(async () => {
  fs.mkdirSync(downloadsPath, {recursive: true});
  let browser, stage = 'open viewer';
  try {
    browser = await chromium.launch({headless: true, downloadsPath,
      ...(process.env.BROWSER_EXECUTABLE ? {executablePath: process.env.BROWSER_EXECUTABLE} : {})});
    const page = await browser.newPage({viewport: {width: 1280, height: 1050}, acceptDownloads: true});
    page.setDefaultTimeout(15000);
    page.on('pageerror', error => validation.javascript_errors.push(String(error)));
    await page.goto(pathToFileURL(path.join(output, 'review.html')).href);
    const clips = await page.locator('#clip option').evaluateAll(options => options.map(option => option.value));
    assert.equal(clips.length, 12, 'Expected twelve excerpt selectors.');
    assert.equal(new Set(clips).size, 12, 'Duplicate excerpt selector.');
    validation.clips = clips.length;
    const durations = [];
    stage = 'verify all excerpts, modes, timelines and downloads';
    for (const clip of clips) {
      await page.selectOption('#clip', clip);
      const modes = await page.locator('#mode option').evaluateAll(options => options.map(option => option.value));
      assert.deepEqual([...modes].sort(), [...expectedModes].sort(), 'Wrong mode inventory for ' + clip);
      for (const mode of expectedModes) {
        await page.selectOption('#mode', mode);
        durations.push(await loadAudioMetadata(page, clip));
        validation.audio_metadata_checks++;
        assert.equal(await page.locator('#roll svg').count(), 1, 'Missing source note plot.');
        assert.equal(await page.locator('#events svg').count(), 1, 'Missing delivery plot.');
        assert.equal(await page.locator('#source-playhead').count(), 1);
        assert.equal(await page.locator('#delivery-playhead').count(), 1);
        assert((await page.locator('#roll svg').getAttribute('aria-label')).includes(clip));
        assert((await page.locator('#events svg').getAttribute('aria-label')).includes(clip + ' · ' + mode));
        assert((await page.locator('#metrics').innerText()).includes('Onset F1'));
        assert((await page.locator('#timing').innerText()).includes('Processing/audio'));
        let noteFilename;
        for (const link of ['json', 'midi', 'csv']) {
          const url = await page.locator('#' + link).evaluate(anchor => anchor.href);
          const filename = savedExport(url);
          assert.equal(path.basename(filename, path.extname(filename)), clip, 'Linked export uses another excerpt.');
          assert.equal(path.basename(path.dirname(filename)), mode, 'Linked export uses another mode.');
          if (link === 'json') noteFilename = filename;
          validation.export_link_checks++;
        }
        const record = JSON.parse(fs.readFileSync(noteFilename.replace(/\.json$/, '-metrics.json'), 'utf8'));
        assert.equal(record.id, clip);
        assert.equal(record.mode, mode);
        await inspectEvents(page, record, mode);
        validation.source_delivery_chart_checks++;
        const exported = JSON.parse(fs.readFileSync(noteFilename, 'utf8'));
        assert(Array.isArray(exported.notes), 'Note export lacks its note array.');
        assert.equal(exported.metadata.mode, mode);
        const downloadEvent = page.waitForEvent('download');
        await page.click('#json');
        const download = await downloadEvent;
        const downloadPath = path.join(downloadsPath, clip + '-' + mode + '.json');
        await download.saveAs(downloadPath);
        assert.deepEqual(JSON.parse(fs.readFileSync(downloadPath, 'utf8')), exported,
          'Blob download differs from its source export; normalize JSON/CRLF semantics.');
        validation.note_json_download_checks++;
        assert.equal(await page.locator('#status').innerText(), '', 'Viewer reports an audio error.');
        validation.combinations++;
      }
      console.log('Viewer verified:', clip, '(4 modes, local audio, event charts and JSON downloads).');
    }
    assert.equal(validation.combinations, 48);
    assert.equal(validation.export_link_checks, 144);
    assert.equal(validation.note_json_download_checks, 48);
    assert.equal(validation.causal_event_log_checks, 36);
    assert(validation.eof_flush_tags_checked > 0, 'No EOF flush tag was exercised.');
    validation.audio_duration_min_seconds = Math.min(...durations);
    validation.audio_duration_max_seconds = Math.max(...durations);
    stage = 'comparison controls preserve playback time';
    await page.selectOption('#clip', clips[0]);
    for (const source of expectedModes.slice(0, 3)) await compare(page, source, 'offline_bidir', 3.25, true);
    await compare(page, 'offline_bidir', 'causal_40ms', 4.25, true);
    await compare(page, 'causal_20ms', 'causal_80ms', 2.75);
    await compare(page, 'causal_80ms', 'causal_40ms', 2.75);
    validation.comparison_button_and_selector_checks = validation.comparison_checks.length;
    stage = 'click source and delivery plots';
    await page.selectOption('#mode', 'causal_40ms');
    await loadAudioMetadata(page, clips[0]);
    let box = await page.locator('#roll svg').boundingBox();
    assert(box, 'Source note plot has no dimensions.');
    await page.locator('#roll svg').click({position: {x: box.width * (45 + 3 / 10 * 945) / 1000, y: box.height / 2}});
    await waitForTime(page, 3);
    box = await page.locator('#events svg').boundingBox();
    const eventDuration = await page.locator('#events svg').evaluate(svg => Number(svg.dataset.duration));
    await page.locator('#events svg').click({position: {x: box.width * (60 + 4 / eventDuration * 925) / 1000, y: box.height / 2}});
    await waitForTime(page, 4);
    validation.source_and_delivery_plot_seek = true;
    stage = 'screenshot and final verification';
    await page.locator('#showall').check();
    await page.screenshot({path: path.join(output, 'review.png'), fullPage: true});
    validation.screenshot = 'review.png';
    assert.deepEqual(validation.javascript_errors, [], 'Viewer emitted JavaScript errors.');
    validation.passed = true;
    console.log('Viewer passed: 48 combinations, 144 exports, 48 exact JSON downloads, event clocks and comparison controls.');
  } catch (error) {
    validation.failed_stage = stage;
    validation.failure = String(error && error.stack || error);
    throw error;
  } finally {
    validation.completed_at = new Date().toISOString();
    fs.mkdirSync(output, {recursive: true});
    fs.writeFileSync(path.join(output, 'viewer-validation.json'), JSON.stringify(validation, null, 2));
    if (browser) await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
