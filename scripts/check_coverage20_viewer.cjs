'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path'), crypto = require('node:crypto');
const {pathToFileURL, fileURLToPath} = require('node:url');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = path.resolve(__dirname, '..', 'outputs', 'coverage20');
const downloadsPath = path.resolve(__dirname, '..', 'tmp', 'browser-downloads', 'coverage20');
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const summary = JSON.parse(fs.readFileSync(path.join(root, 'summary.json'), 'utf8'));
const notes = JSON.parse(fs.readFileSync(path.join(root, 'reference-notes.json'), 'utf8'));
const result = {passed:false, model_policy_views:0, filter_states:0, exact_json_downloads:0,
  export_links:0, source_case_links:0, javascript_errors:[]};
(async () => {
  let browser;
  try {
    fs.mkdirSync(downloadsPath, {recursive:true});
    browser = await chromium.launch({headless:true, downloadsPath,
      ...(process.env.BROWSER_EXECUTABLE ? {executablePath:process.env.BROWSER_EXECUTABLE} : {})});
    const page = await browser.newPage({viewport:{width:1280,height:1000}, acceptDownloads:true});
    page.on('pageerror', error => result.javascript_errors.push(String(error)));
    await page.goto(pathToFileURL(path.join(root, 'review.html')).href);
    assert((await page.locator('h1').innerText()).includes('fret coverage'));
    for (const [mode, model] of Object.entries(summary.modes)) {
      await page.selectOption('#model', mode);
      for (const [policy, detail] of Object.entries(model.policies)) {
        await page.selectOption('#policy', policy);
        assert.deepEqual(await page.evaluate(() => current.detail), detail);
        assert.equal(await page.locator('#bins-table tbody tr').count(), 4);
        const unseen = await page.locator('#bins-table tbody tr').first().innerText();
        assert(unseen.includes('unseen') && unseen.includes('—'));
        assert((await page.locator('#false-activity').innerText()).includes(detail.overall.reference_inactive_frames.toLocaleString('en-US')));
        const exports = await page.locator('.exports a').evaluateAll(nodes => nodes.map(n => n.href));
        assert.equal(exports.length, 9);
        for (const url of exports) {
          const file = fileURLToPath(url), rel = path.relative(root,file);
          assert(!rel.startsWith('..') && !path.isAbsolute(rel) && fs.statSync(file).size > 0);
          result.export_links++;
        }
        const pending = page.waitForEvent('download');
        await page.click('#download-notes');
        const download = await pending;
        assert.equal(await download.failure(), null);
        assert.equal(hash(await download.path()), hash(path.join(root,'reference-notes.json')));
        result.exact_json_downloads++;
        for (const bin of ['all', ...summary.bins]) {
          await page.selectOption('#bin', bin);
          for (const string of ['all','0','1','2','3','4','5']) {
            await page.selectOption('#string', string);
            for (const all of [false,true]) {
              await page.locator('#all-positions').setChecked(all);
              const expected = detail.classes.filter(r => (bin==='all'||r.bin===bin)
                && (string==='all'||r.string===Number(string)) && (all||r.reference_active||r.reference_notes));
              assert.deepEqual(await page.evaluate(() => current.rows), expected);
              assert.equal(await page.locator('#classes-table tbody tr').count(), expected.length);
              const filtered = notes.filter(r=>r.mode===mode&&(bin==='all'||r.bin===bin)&&(string==='all'||r.string===Number(string)));
              assert((await page.locator('#note-count').innerText()).startsWith(filtered.length+' reference notes;'));
              assert.equal(await page.locator('#notes-table tbody tr').count(), Math.min(20,filtered.length));
              result.filter_states++;
            }
          }
        }
        result.model_policy_views++;
      }
    }
    const sourceCases = [...new Set(notes.map(n => n.mode+'/'+n.id))];
    assert.equal(sourceCases.length,36);
    for (const item of sourceCases) {
      const [mode,id] = item.split('/');
      const url = new URL('../fret19/validation/results/'+encodeURIComponent(mode)+'/'+encodeURIComponent(id)+'-metrics.json',page.url());
      assert.equal(url.hash,'');
      assert(fs.statSync(fileURLToPath(url)).size > 0);
      if (id.includes('#')) assert(url.href.includes('%23'));
      result.source_case_links++;
    }
    await page.selectOption('#model','control500');
    await page.selectOption('#policy','argmax');
    await page.selectOption('#bin','all');
    await page.selectOption('#string','all');
    await page.locator('#all-positions').uncheck();
    await page.screenshot({path:path.join(root,'review.png'),fullPage:true});
    await page.setViewportSize({width:420,height:950});
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1));
    await page.screenshot({path:path.join(root,'review-mobile.png'),fullPage:true});
    assert.equal(result.javascript_errors.length,0);
    result.passed=true;
    result.review_html_sha256=hash(path.join(root,'review.html'));
    result.browser_source_sha256=hash(__filename);
    result.screenshot_sha256=hash(path.join(root,'review.png'));
    result.mobile_screenshot_sha256=hash(path.join(root,'review-mobile.png'));
  } catch (error) {
    result.error=String(error.stack||error); process.exitCode=1;
  } finally {
    if(browser) await browser.close();
    fs.writeFileSync(path.join(root,'viewer-validation.json'),JSON.stringify(result,null,2)+'\n');
  }
  console.log(JSON.stringify(result));
})();
