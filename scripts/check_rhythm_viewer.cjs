// Local headless UI checks. Use the bundled Playwright path, or set PLAYWRIGHT_MODULE.
const fs = require('fs');
const path = require('path');
const {pathToFileURL} = require('url');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
(async()=>{
  fs.mkdirSync('tmp/browser-downloads',{recursive:true});
  const browser = await chromium.launch({headless:true,downloadsPath:path.resolve('tmp/browser-downloads'), ...(process.env.BROWSER_EXECUTABLE ? {executablePath:process.env.BROWSER_EXECUTABLE}: {})});
  try {
    const page = await browser.newPage({viewport:{width:1240,height:1000},acceptDownloads:true});
    const errors = []; page.on('pageerror',e=>errors.push(String(e)));
    await page.goto(pathToFileURL(path.resolve('outputs/rhythm/review.html')).href);
    await page.waitForFunction(()=>Number.isFinite(document.querySelector('audio').duration));
    const clips = await page.locator('#clip option').count(), scenarios = await page.locator('#scenario option').count();
    if(clips!==12 || scenarios!==4) throw Error('Wrong excerpt/scenario inventory.');
    for(let c=0;c<clips;c++) for(let s=0;s<scenarios;s++) {
      await page.selectOption('#clip',{index:c}); await page.selectOption('#scenario',{index:s});
      if(await page.locator('#bars .bar').count() < 1 || !(await page.locator('#metrics').innerText()).includes('notes')) throw Error('Missing bars/metrics.');
    }
    await page.selectOption('#clip',{index:0}); await page.selectOption('#scenario','predicted_notes_predicted_beats');
    await page.locator('#bars .bar').nth(1).click();
    await page.waitForFunction(()=>document.querySelector('audio').currentTime>0);
    await page.waitForFunction(()=>document.querySelectorAll('#bars .active').length===1);
    await page.screenshot({path:'outputs/rhythm/review.png',fullPage:true});
    const firstDownload=page.waitForEvent('download'); await page.click('#raw');
    await (await firstDownload).saveAs('outputs/rhythm/example/editor-raw.json');
    await page.fill('#num','4'); await page.selectOption('#den','4'); await page.fill('#pulse','1');
    const secondDownload=page.waitForEvent('download'); await page.click('#download');
    await (await secondDownload).saveAs('outputs/rhythm/example/editor-corrections.json');
    const correction=JSON.parse(fs.readFileSync('outputs/rhythm/example/editor-corrections.json','utf8'));
    if(correction.beats.length<2||correction.meter.numerator!==4) throw Error('Correction download failed.');
    await page.fill('#beats','0, 0'); await page.click('#download');
    if(!(await page.locator('#status').innerText()).includes('strictly increasing')) throw Error('Invalid grid accepted.');
    if(errors.length) throw Error(errors.join('\n'));
    fs.writeFileSync('outputs/rhythm/viewer-validation.json',JSON.stringify({clips,scenarios,combinations:clips*scenarios,
      audio_metadata_loaded:true,seek_and_bar_highlight:true,raw_notes_download:true,corrections_download:true,
      invalid_grid_rejected:true,javascript_errors:errors},null,2));
    console.log('Viewer passed: 48 clip/scenario combinations, audio seek, exports and correction validation.');
  } finally { await browser.close(); }
})().catch(e=>{console.error(e);process.exitCode=1;});
