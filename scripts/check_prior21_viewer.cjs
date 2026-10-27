'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const {pathToFileURL,fileURLToPath}=require('node:url');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const root=path.resolve(__dirname,'..','outputs','prior21');
const downloadsPath=path.resolve(__dirname,'..','tmp','browser-downloads','prior21');
const hash=file=>crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const summary=JSON.parse(fs.readFileSync(path.join(root,'summary.json'),'utf8'));
const matrices=JSON.parse(fs.readFileSync(path.join(root,'confusions.json'),'utf8'));
const result={passed:false,model_predictor_views:0,filter_states:0,exact_json_downloads:0,export_links:0,source_case_links:0,javascript_errors:[]};
(async()=>{let browser;try{
 fs.mkdirSync(downloadsPath,{recursive:true});browser=await chromium.launch({headless:true,downloadsPath,...(process.env.BROWSER_EXECUTABLE?{executablePath:process.env.BROWSER_EXECUTABLE}:{})});
 const page=await browser.newPage({viewport:{width:1280,height:1000},acceptDownloads:true});page.on('pageerror',error=>result.javascript_errors.push(String(error)));
 await page.goto(pathToFileURL(path.join(root,'review.html')).href);assert((await page.locator('h1').innerText()).includes('beat frequency'));
 for(const [mode,model]of Object.entries(summary.modes)){
  await page.selectOption('#model',mode);
  const pending=page.waitForEvent('download');await page.click('#download-confusions');const download=await pending;
  assert.equal(await download.failure(),null);assert.equal(hash(await download.path()),hash(path.join(root,'confusions.json')));result.exact_json_downloads++;
  const sourceLinks=await page.locator('#clips a').evaluateAll(nodes=>nodes.map(n=>n.href));assert.equal(sourceLinks.length,12);
  for(const url of sourceLinks){assert.equal(new URL(url).hash,'');assert(fs.statSync(fileURLToPath(url)).size>0);if(fileURLToPath(url).includes('#'))assert(url.includes('%23'));result.source_case_links++;}
  for(const [predictor,detail]of Object.entries(model.predictors)){
   await page.selectOption('#predictor',predictor);assert.deepEqual(await page.evaluate(()=>current.detail),detail);
   assert.equal(await page.locator('#scores tbody tr').count(),8);
   const exports=await page.locator('.exports a').evaluateAll(nodes=>nodes.map(n=>n.href));assert.equal(exports.length,11);
   for(const url of exports){const file=fileURLToPath(url),relative=path.relative(root,file);assert(!relative.startsWith('..')&&!path.isAbsolute(relative)&&fs.statSync(file).size>0);result.export_links++;}
   for(let string=0;string<6;string++){
    await page.selectOption('#string',String(string));
    for(const normalize of [true,false]){await page.locator('#normalize').setChecked(normalize);
     for(const all of [false,true]){await page.locator('#all-rows').setChecked(all);
      const rows=detail.classes.filter(r=>r.string===string&&(all||r.reference_active_frames));
      assert.deepEqual(await page.evaluate(()=>current.rows),rows);assert.deepEqual(await page.evaluate(()=>current.matrix),matrices[mode][predictor][string]);
      assert.equal(await page.locator('#heatmap tbody tr').count(),rows.length);assert.equal(await page.locator('#classes tbody tr').count(),rows.length);
      const cells=await page.locator('#heatmap tbody td').evaluateAll(nodes=>nodes.map(n=>({gt:Number(n.dataset.gt),pred:Number(n.dataset.pred),value:Number(n.dataset.value),text:n.textContent})));
      assert.equal(cells.length,rows.length*25);
      for(const cell of cells){assert.equal(cell.value,matrices[mode][predictor][string][cell.gt][cell.pred]);if(cell.value===0)assert.equal(cell.text,'');}
      result.filter_states++;
     }
    }
   }
   result.model_predictor_views++;
  }
 }
 await page.selectOption('#model','control500');await page.selectOption('#predictor','model_best_active');await page.selectOption('#string','0');await page.locator('#normalize').check();await page.locator('#all-rows').uncheck();
 await page.screenshot({path:path.join(root,'review.png'),fullPage:true});await page.setViewportSize({width:420,height:950});
 assert(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth+1));await page.screenshot({path:path.join(root,'review-mobile.png'),fullPage:true});
 assert.equal(result.javascript_errors.length,0);result.passed=true;result.review_html_sha256=hash(path.join(root,'review.html'));result.browser_source_sha256=hash(__filename);
 result.screenshot_sha256=hash(path.join(root,'review.png'));result.mobile_screenshot_sha256=hash(path.join(root,'review-mobile.png'));
}catch(error){result.error=String(error.stack||error);process.exitCode=1;}finally{if(browser)await browser.close();fs.writeFileSync(path.join(root,'viewer-validation.json'),JSON.stringify(result,null,2)+'\n');}console.log(JSON.stringify(result));})();
