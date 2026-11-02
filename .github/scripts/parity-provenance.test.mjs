import assert from 'node:assert/strict';
import { copyFileSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { test } from 'node:test';
import process from 'node:process';
import { URL, pathToFileURL } from 'node:url';

const sha = '0123456789abcdef0123456789abcdef01234567';
const merged = { status: 'ahead', base_commit: { sha }, merge_base_commit: { sha } };
const unmerged = { status: 'behind', base_commit: { sha }, merge_base_commit: { sha } };
const branch = [{ name: 'reviewed-change', commit: { sha } }];

function execute(t, scenario, release = false) {
  const root = mkdtempSync(join(tmpdir(), 'prism-parity-provenance-'));
  t.after(() => { assert(root.startsWith(join(tmpdir(), 'prism-parity-provenance-'))); rmSync(root, { recursive: true, force: true }); });
  const scripts = join(root, '.github/scripts'), output = join(root, 'output'), mock = join(root, 'mock.mjs');
  mkdirSync(scripts, { recursive: true }); mkdirSync(join(root, 'conformance'));
  copyFileSync(new URL('./parity-pin.mjs', import.meta.url), join(scripts, 'parity-pin.mjs'));
  writeFileSync(join(root, 'conformance/parity.lock'), sha + '\n');
  writeFileSync(mock, `import cp from 'node:child_process';
import { syncBuiltinESMExports } from 'node:module';
const data = ${JSON.stringify(scenario)};
cp.execFileSync = (file, args) => {
  if (file !== 'gh' || args[0] !== 'api' || args[1] !== '--method' || args[2] !== 'GET') throw new Error('Unexpected command');
  const endpoint = args.at(-1);
  if (data.apiError) throw new Error('synthetic-api-secret');
  if (endpoint === 'repos/Particle-Academy/prism-parity/commits/${sha}/branches-where-head') return JSON.stringify(data.branches);
  if (endpoint === 'repos/Particle-Academy/prism-parity/compare/${sha}...main') return JSON.stringify(data.comparison);
  throw new Error('Wrong API subject');
};
syncBuiltinESMExports();
`);
  const result = spawnSync(process.execPath, ['--import', pathToFileURL(mock).href, join(scripts, 'parity-pin.mjs'), ...(release ? ['--release'] : [])], { cwd: root, env: { ...process.env, GITHUB_OUTPUT: output, NODE_OPTIONS: '', NODE_PATH: '' }, encoding: 'utf8', timeout: 10_000 });
  assert.doesNotMatch(result.stderr, /synthetic-api-secret/);
  return { ...result, output };
}

for (const [name, scenario] of Object.entries({
  'canonical branch head': { branches: branch, comparison: unmerged },
  'merged commit no longer at a branch head': { branches: [], comparison: merged },
})) {
  test('provenance accepts ' + name, (t) => {
    const result = execute(t, scenario);
    assert.equal(result.status, 0, result.stderr);
    assert.equal(result.stdout, sha + '\n');
    assert.equal(readFileSync(result.output, 'utf8'), 'sha=' + sha + '\n');
  });
}

for (const [name, scenario] of Object.entries({
  'fork-only commit': { branches: [], comparison: unmerged },
  'unknown or API failure': { apiError: true },
  'malformed branch evidence': { branches: {}, comparison: merged },
  'wrong branch-head subject': { branches: [{ name: 'other', commit: { sha: 'a'.repeat(40) } }], comparison: unmerged },
})) {
  test('provenance refuses ' + name + ' before output', (t) => {
    const result = execute(t, scenario);
    assert.equal(result.status, 1);
    assert.equal(result.stdout, '');
    assert.equal(existsSync(result.output), false);
  });
}

test('release refuses even an owned branch head until it is merged', (t) => {
  const result = execute(t, { branches: branch, comparison: unmerged }, true);
  assert.equal(result.status, 1);
  assert.match(result.stderr, /not merged/);
  assert.equal(existsSync(result.output), false);
});

test('release still accepts a merged pin', (t) => {
  const result = execute(t, { branches: [], comparison: merged }, true);
  assert.equal(result.status, 0, result.stderr);
  assert.equal(existsSync(result.output), false);
});

test('canonical execution jobs have no dependency cache and every pin read is authenticated', () => {
  let checked = 0;
  for (const name of ['conformance', 'factcheck', 'static', 'tests', 'parity-drift']) {
    const text = readFileSync(new URL('../workflows/' + name + '.yml', import.meta.url), 'utf8');
    for (const job of text.split(/^ {2}(?=[a-zA-Z0-9_-]+:\s*$)/m)) {
      if (!job.includes('repository: Particle-Academy/prism-parity')) continue;
      checked++;
      assert.doesNotMatch(job, /^\s+cache:|uses: actions\/cache@/m, name);
      if (name !== 'parity-drift') assert.match(job, /id: parity\n {8}env:\n {10}GH_TOKEN: \$\{\{ github.token \}\}/, name);
    }
  }
  assert.equal(checked, 5);
});


test('project pin helpers run only after the verified dependency policy gate', () => {
  const conformance = readFileSync(new URL('../workflows/conformance.yml', import.meta.url), 'utf8');
  const publish = readFileSync(new URL('../workflows/publish.yml', import.meta.url), 'utf8');
  const gate = text => text.indexOf('- name: Third-party dependency allowlist');
  assert(gate(conformance) >= 0 && gate(publish) >= 0);
  assert.doesNotMatch(conformance.slice(0, gate(conformance)), /run:\s*node\s+(?!-e\b)/);
  assert.doesNotMatch(publish.slice(0, gate(publish)), /run:\s*node\s+(?!-e\b)/);
  assert(conformance.indexOf('- name: Read pinned parity revision') > gate(conformance));
  assert(conformance.indexOf('- name: Checkout prism-parity') > conformance.indexOf('- name: Read pinned parity revision'));
  assert(publish.indexOf('- name: Pinned corpus must be merged') > gate(publish));
  assert(publish.indexOf('run: node .github/scripts/parity-pin.mjs --release') < publish.indexOf('\n  publish:'));
});
