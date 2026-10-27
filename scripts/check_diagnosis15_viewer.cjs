'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const {pathToFileURL,fileURLToPath}=require('node:url');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const root=path.resolve(__dirname,'..','outputs','diagnosis15');
const downloadsPath=path.resolve(__dirname,'..','tmp','browser-downloads','diagnosis15');
const hash=file=>crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const validation={passed:false,views:0,export_links:0,exact_json_downloads:0,category_filters:0,
  reference_rows_checked:0,paired_switches:0,note_seek_checks:0,javascript_errors:[]};

function localFile(url){const file=fileURLToPath(url),relative=path.relative(root,file);assert(relative&&!relative.startsWith('..')&&!path.isAbsolute(relative));assert(fs.existsSync(file)&&fs.statSync(file).size>0);const real=path.relative(fs.realpathSync(root),fs.realpathSync(file));assert(!real.startsWith('..')&&!path.isAbsolute(real));return file;}
async function seek(page,time){await page.locator('#audio').evaluate((a,t)=>{a.currentTime=t;},time);await page.waitForFunction(t=>!document.getElementById('audio').seeking&&Math.abs(document.getElementById('audio').currentTime-t)<.03&&Math.abs(parseFloat(document.getElementById('clock').textContent)-t)<.03,time);}

(async()=>{let browser;try{
 validation.review_html_sha256=hash(path.join(root,'review.html'));validation.browser_check_source_sha256=hash(__filename);
 fs.mkdirSync(downloadsPath,{recursive:true});browser=await chromium.launch({headless:true,downloadsPath,...(process.env.BROWSER_EXECUTABLE?{executablePath:process.env.BROWSER_EXECUTABLE}:{})});
 const page=await browser.newPage({viewport:{width:1280,height:1000},acceptDownloads:true});page.on('pageerror',error=>validation.javascript_errors.push(String(error)));
 await page.goto(pathToFileURL(path.join(root,'review.html')).href);assert((await page.locator('h1').innerText()).includes('notes going wrong'));
 const clips=await page.locator('#clip option').evaluateAll(nodes=>nodes.map(n=>n.value)),models=await page.locator('#model option').evaluateAll(nodes=>nodes.map(n=>n.value)),filters=await page.locator('#filter option').evaluateAll(nodes=>nodes.map(n=>n.value));
 assert.equal(clips.length,12);assert.equal(models.length,2);assert.equal(filters.length,7);
 for(const clip of clips){await page.selectOption('#clip',clip);await page.waitForFunction(()=>document.getElementById('audio').readyState>=1&&Math.abs(document.getElementById('audio').duration-12)<.01);localFile(await page.locator('#audio').evaluate(a=>a.src));
  for(const model of models){await page.selectOption('#model',model);await page.selectOption('#filter','all');
   const stem=path.join(root,'validation','results',model,clip),row=JSON.parse(fs.readFileSync(stem+'.json','utf8'));
   const shown=await page.evaluate(()=>({id:current.id,model:current.model,notes:current.notes,reference_rows:current.reference_rows,categories:current.categories,f1:current.f1}));
   assert.equal(shown.id,clip);assert.equal(shown.model,model);assert.deepEqual(shown.notes,row.notes);assert.deepEqual(shown.reference_rows,row.taxonomy.reference_rows);assert.deepEqual(shown.categories,row.taxonomy.reference_counts);assert.equal(shown.f1,row.pitch_metrics.onset.f1);
   validation.reference_rows_checked+=shown.reference_rows.length;
   assert.equal(await page.locator('#noterows tr').count(),shown.reference_rows.length);
   const links=await page.locator('#exports a').evaluateAll(nodes=>nodes.map(n=>({name:n.textContent,url:n.href})));assert.equal(links.length,5);
   for(const link of links){localFile(link.url);validation.export_links++;}
   const exact=links.find(link=>link.name==='Diagnosis JSON'),notesLink=links.find(link=>link.name==='Notes JSON');assert.deepEqual(JSON.parse(fs.readFileSync(localFile(notesLink.url),'utf8')).notes,row.notes);
   const pending=page.waitForEvent('download');await page.click('#download-json');const download=await pending;assert.equal(await download.failure(),null);assert.equal(hash(await download.path()),hash(fileURLToPath(exact.url)));validation.exact_json_downloads++;
   for(const filter of filters.slice(1)){await page.selectOption('#filter',filter);const expected=row.taxonomy.reference_rows.filter(r=>r.category===filter);const actual=await page.locator('#noterows tr').evaluateAll(nodes=>nodes.map(n=>({index:Number(n.dataset.referenceIndex),category:n.dataset.category,cells:[...n.cells].map(c=>c.textContent)})));assert.equal(actual.length,expected.length);assert.deepEqual(actual.map(r=>r.index),expected.map(r=>r.reference_index));
    for(let i=0;i<actual.length;i++){assert.equal(actual[i].category,filter);const evidence=expected[i].evidence;for(const [index,key]of [[4,'same_string_fret_fraction'],[5,'any_string_pitch_fraction'],[6,'auxiliary_pitch_fraction']])assert.equal(actual[i].cells[index],evidence[key]===null?'—':(100*evidence[key]).toFixed(0)+'%');}
    assert((await page.locator('#rowcount').innerText()).startsWith(expected.length+' of '));validation.category_filters++;
   }
   await page.selectOption('#filter','all');if(row.taxonomy.reference_rows.length){await page.locator('#noterows button').first().click();const target=row.taxonomy.reference_rows[0].note.start;await page.waitForFunction(t=>!document.getElementById('audio').seeking&&Math.abs(document.getElementById('audio').currentTime-t)<.03,target);validation.note_seek_checks++;}
   await seek(page,5);const url=await page.locator('#audio').evaluate(a=>a.currentSrc);await page.click('#compare');assert.equal(await page.locator('#model').inputValue(),models.find(m=>m!==model));assert.equal(await page.locator('#audio').evaluate(a=>a.currentSrc),url);assert(Math.abs(await page.locator('#audio').evaluate(a=>a.currentTime)-5)<.03);assert(Math.abs(Number(await page.locator('#source-playhead').getAttribute('x1'))-517.5)<.1);validation.paired_switches++;validation.views++;
  }
 }
 await page.selectOption('#clip',clips[0]);await page.selectOption('#model','group8_500');await page.selectOption('#filter','all');await page.locator('#audio').evaluate(a=>{a.pause();a.currentTime=0;});await page.screenshot({path:path.join(root,'review.png'),fullPage:true});
 assert.equal(validation.javascript_errors.length,0);validation.passed=true;validation.screenshot_sha256=hash(path.join(root,'review.png'));
}catch(error){validation.error=String(error.stack||error);process.exitCode=1;}finally{if(browser)await browser.close();fs.writeFileSync(path.join(root,'viewer-validation.json'),JSON.stringify(validation,null,2)+'\n');}console.log(JSON.stringify(validation));})();
