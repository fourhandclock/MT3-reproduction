'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const {pathToFileURL, fileURLToPath} = require('node:url');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = path.resolve(__dirname, '..', 'outputs', 'decoder14');
const downloadsPath = path.resolve(__dirname, '..', 'tmp', 'browser-downloads', 'decoder14');
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const validation = {passed:false, views:0, exports:0, exact_json_downloads:0,
  packet_event_checks:0, comparison_switches:0, causal_log_checks:0, javascript_errors:[]};

function localFile(url) {
  const filename = fileURLToPath(url);
  const relative = path.relative(root, filename);
  assert(relative && !relative.startsWith('..') && !path.isAbsolute(relative));
  assert(fs.existsSync(filename) && fs.statSync(filename).size > 0);
  const resolved = path.relative(fs.realpathSync(root), fs.realpathSync(filename));
  assert(!resolved.startsWith('..') && !path.isAbsolute(resolved));
  return filename;
}

async function seek(page, seconds) {
  await page.locator('#audio').evaluate((audio,t)=>{audio.currentTime=t;},seconds);
  await page.waitForFunction(t=>Math.abs(document.getElementById('audio').currentTime-t)<.03 &&
    !document.getElementById('audio').seeking &&
    Math.abs(parseFloat(document.getElementById('clock').textContent)-t)<.03,seconds);
}

(async()=>{
  let browser;
  try {
    validation.review_html_sha256=hash(path.join(root,'review.html'));
    validation.browser_check_source_sha256=hash(__filename);
    fs.mkdirSync(downloadsPath,{recursive:true});
    browser=await chromium.launch({headless:true,downloadsPath,
      ...(process.env.BROWSER_EXECUTABLE?{executablePath:process.env.BROWSER_EXECUTABLE}:{})});
    const page=await browser.newPage({viewport:{width:1280,height:1000},acceptDownloads:true});
    page.on('pageerror',error=>validation.javascript_errors.push(String(error)));
    await page.goto(pathToFileURL(path.join(root,'review.html')).href);
    assert((await page.locator('h1').innerText()).includes('Frozen decoder calibration'));
    const modes=await page.locator('#mode option').evaluateAll(nodes=>nodes.map(n=>n.value));
    const clips=await page.locator('#clip option').evaluateAll(nodes=>nodes.map(n=>n.value));
    assert.equal(clips.length,12);assert.equal(modes.length,4);
    for(const clip of clips){
      await page.selectOption('#clip',clip);
      await page.waitForFunction(()=>document.getElementById('audio').readyState>=1 &&
        Math.abs(document.getElementById('audio').duration-12)<.01);
      localFile(await page.locator('#audio').evaluate(a=>a.src));
      for(const mode of modes){
        await page.selectOption('#mode',mode);
        const stem=path.join(root,'validation','results',mode,clip);
        const row=JSON.parse(fs.readFileSync(stem+'-metrics.json','utf8'));
        const shown=await page.evaluate(()=>({id:current.id,mode:current.mode,notes:current.notes,
          events:current.events,parameters:current.parameters,f1:current.f1}));
        assert.equal(shown.id,clip);assert.equal(shown.mode,mode);
        assert.deepEqual(shown.notes,row.notes);assert.deepEqual(shown.events,row.events);
        assert.deepEqual(shown.parameters,row.parameters);assert.equal(shown.f1,row.pitch_metrics.onset.f1);
        const links=await page.locator('#exports a').evaluateAll(nodes=>nodes.map(n=>({name:n.textContent,url:n.href})));
        assert.equal(links.length,5);
        for(const link of links){const filename=localFile(link.url);assert(path.basename(filename).startsWith(clip));validation.exports++;}
        const jsonLink=links.find(link=>link.name==='JSON');
        const downloadPromise=page.waitForEvent('download');
        await page.click('#download-json');const download=await downloadPromise;
        assert.equal(await download.failure(),null);
        assert.equal(hash(await download.path()),hash(fileURLToPath(jsonLink.url)));
        validation.exact_json_downloads++;
        const marks=await page.locator('#events .event').evaluateAll(nodes=>nodes.map(n=>({
          source:Number(n.dataset.source),packet:Number(n.dataset.packet),title:n.querySelector('title').textContent,
          circles:[...n.querySelectorAll('circle')].map(c=>Number(c.getAttribute('cx')))})));
        assert.equal(marks.length,row.events.length);
        for(let i=0;i<marks.length;i++){
          const m=marks[i],e=row.events[i];
          assert.equal(m.source,e.source_time);assert.equal(m.packet,e.emitted_at_audio_seconds);
          assert(!('delivered_at_seconds' in e));assert(m.packet>=m.source-1e-9);
          assert(Math.abs(m.circles[0]-(45+e.source_time/12*945))<1e-8);
          assert(Math.abs(m.circles[1]-(45+e.emitted_at_audio_seconds/12*945))<1e-8);
          if(e.flush)assert(m.title.includes('(flush)'));
          validation.packet_event_checks++;
        }
        await seek(page,5);
        const count=row.events.filter(e=>e.emitted_at_audio_seconds<=5).length;
        assert((await page.locator('#eventcount').innerText()).startsWith(count+' of '));
        assert.equal(await page.locator('#eventrows tr').count(),Math.min(40,count));
        const times=await page.locator('#eventrows tr').evaluateAll(nodes=>nodes.map(n=>Number(n.cells[4].textContent)));
        assert(times.every(t=>t<=5));validation.causal_log_checks++;
        assert(Math.abs(Number(await page.locator('#source-playhead').getAttribute('x1'))-517.5)<.1);
        assert(Math.abs(Number(await page.locator('#packet-playhead').getAttribute('x1'))-438.75)<.1);
        await page.locator('#showall').check();
        assert.equal(await page.locator('#eventrows tr').count(),Math.min(40,row.events.length));
        await page.locator('#showall').uncheck();
        const target=mode.endsWith('_strict')?mode.replace(/_strict$/,'_selected'):mode.replace(/_selected$/,'_strict');
        const audioUrl=await page.locator('#audio').evaluate(a=>a.currentSrc);
        await page.click('#compare');
        assert.equal(await page.locator('#mode').inputValue(),target);
        assert.equal(await page.locator('#audio').evaluate(a=>a.currentSrc),audioUrl);
        assert(Math.abs(await page.locator('#audio').evaluate(a=>a.currentTime)-5)<.03);
        validation.comparison_switches++;validation.views++;
      }
    }
    await page.selectOption('#clip',clips[0]);await page.selectOption('#mode','group8_500_selected');
    await page.locator('#audio').evaluate(a=>{a.pause();a.currentTime=0;});
    await page.screenshot({path:path.join(root,'review.png'),fullPage:true});
    assert.equal(validation.javascript_errors.length,0);
    validation.passed=true;
    validation.screenshot_sha256=hash(path.join(root,'review.png'));
  }catch(error){validation.error=String(error.stack||error);process.exitCode=1;}
  finally{if(browser)await browser.close();fs.writeFileSync(path.join(root,'viewer-validation.json'),JSON.stringify(validation,null,2)+'\n');}
  console.log(JSON.stringify(validation));
})();
