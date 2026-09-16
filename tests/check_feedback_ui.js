// Exercise save/correction/error states without a browser dependency.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
class Element {
  constructor() { this.children = []; this.dataset = {}; this.attributes = {}; this.handlers = {}; this.classList = {remove(){}, toggle(){}}; }
  append(...items) { this.children.push(...items); }
  after(item) { this.adjacent = item; }
  setAttribute(key, value) { this.attributes[key] = value; }
  addEventListener(name, handler) { this.handlers[name] = handler; }
}
const elements = Object.fromEntries(['search-form', 'query', 'submit', 'result'].map(id => [id, new Element()]));
let ok = true;
const context = vm.createContext({ document: {getElementById: id => elements[id], createElement: () => new Element()},
  fetch: async () => ({ok, json: async () => ok ? {saved: true} : {error: 'Database unavailable'} }) });
vm.runInContext(fs.readFileSync('frontend/app.js', 'utf8'), context);
vm.runInContext('showResults([{id:1,title:"One"},{id:2,title:"Two"}], "test-search")', context);
const list = elements.result.children.find(el => el.className === 'hits');
const first = list.children[0].children[2];
const second = list.children[1].children[2];
(async () => {
  await first.handlers.click();
  assert.equal(first.textContent, 'Saved ✓');
  assert.equal(first.attributes['aria-pressed'], 'true');
  await second.handlers.click();
  assert.equal(first.textContent, 'This is it');
  assert.equal(second.textContent, 'Saved ✓');
  ok = false;
  await first.handlers.click();
  assert.equal(first.textContent, 'This is it');
  assert.match(first.adjacent.textContent, /Feedback not saved: Database unavailable/);
  assert.equal(first.disabled, false);
  console.log('UI confirmation, correction and visible error states passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
