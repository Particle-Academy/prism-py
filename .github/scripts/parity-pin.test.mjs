import assert from 'node:assert/strict';
import { copyFileSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
import { test } from 'node:test';
import process from 'node:process';
import { URL, pathToFileURL } from 'node:url';
import { readPin, assertMerged } from './parity-pin.mjs';

const sha = '0123456789abcdef0123456789abcdef01234567';
function fixture(t) {
  const root = mkdtempSync(join(tmpdir(), 'prism-parity-pin-'));
  t.after(() => { assert(root.startsWith(join(tmpdir(), 'prism-parity-pin-'))); rmSync(root, { recursive: true, force: true }); });
  return root;
}

test('accepts only a complete commit hash with an optional single line ending', (t) => {
  const file = join(fixture(t), 'parity.lock');
  for (const text of [sha, sha + '\n', sha + '\r\n', sha.toUpperCase()]) {
    writeFileSync(file, text);
    assert.equal(readPin(file), sha);
  }
});

for (const [name, text] of Object.entries({ missing: null, empty: '', branch: 'main\n', short: sha.slice(1), long: sha + 'a', whitespace: ' ' + sha, twoLines: sha + '\n' + sha, blankLine: sha + '\n\n', nonHex: 'g'.repeat(40), injection: sha + '\nsha=main\n' })) {
  test('refuses ' + name + ' before emitting an Actions output', (t) => {
    const root = fixture(t), scripts = join(root, '.github/scripts'), conformance = join(root, 'conformance'), output = join(root, 'output');
    mkdirSync(scripts, { recursive: true }); mkdirSync(conformance);
    copyFileSync(new URL('./parity-pin.mjs', import.meta.url), join(scripts, 'parity-pin.mjs'));
    if (text !== null) writeFileSync(join(conformance, 'parity.lock'), text);
    const result = spawnSync(process.execPath, [join(scripts, 'parity-pin.mjs')], { cwd: root, env: { ...process.env, GITHUB_OUTPUT: output, NODE_OPTIONS: '', NODE_PATH: '' }, encoding: 'utf8', timeout: 10_000 });
    assert.equal(result.status, 1); assert.match(result.stderr, /parity.lock/); assert.equal(result.stdout, ''); assert.equal(existsSync(output), false);
  });
}

test('refuses non-files and symbolic links', (t) => {
  const root = fixture(t), directory = join(root, 'directory'); mkdirSync(directory);
  assert.throws(() => readPin(directory), /parity.lock/);
  if (process.platform !== 'win32') {
    const file = join(root, 'target'), link = join(root, 'link'); writeFileSync(file, sha); symlinkSync(file, link);
    assert.throws(() => readPin(link), /parity.lock/);
  }
});

for (const status of ['ahead', 'identical']) {
  test('release accepts a merged pin: ' + status, () => {
    assert.doesNotThrow(() => assertMerged(sha, { status, base_commit: { sha }, merge_base_commit: { sha } }));
  });
}
for (const status of ['behind', 'diverged', undefined, 'unexpected']) {
  test('release refuses a non-ancestor or unknown comparison: ' + status, () => {
    assert.throws(() => assertMerged(sha, { status, base_commit: { sha }, merge_base_commit: { sha } }), /not merged/);
  });
}
test('release refuses wrong subjects, absent evidence and malformed responses', () => {
  for (const comparison of [null, {}, { status: 'ahead' }, { status: 'ahead', base_commit: { sha: 'a'.repeat(40) }, merge_base_commit: { sha } }, { status: 'ahead', base_commit: { sha }, merge_base_commit: { sha: 'b'.repeat(40) } }]) {
    assert.throws(() => assertMerged(sha, comparison), /not merged/);
  }
});

test('valid CLI output contains exactly the validated commit', (t) => {
  const root = fixture(t), scripts = join(root, '.github/scripts'), output = join(root, 'output');
  mkdirSync(scripts, { recursive: true }); mkdirSync(join(root, 'conformance'));
  copyFileSync(new URL('./parity-pin.mjs', import.meta.url), join(scripts, 'parity-pin.mjs'));
  writeFileSync(join(root, 'conformance/parity.lock'), sha + '\n');
  const mock = join(root, 'mock.mjs');
  writeFileSync(mock, `import cp from 'node:child_process';
import { syncBuiltinESMExports } from 'node:module';
cp.execFileSync = (file, args) => {
  if (file !== 'gh' || args.at(-1) !== 'repos/Particle-Academy/prism-parity/commits/${sha}/branches-where-head') throw new Error('Wrong API subject');
  return JSON.stringify([{ name: 'owned', commit: { sha: '${sha}' } }]);
};
syncBuiltinESMExports();
`);
  const result = spawnSync(process.execPath, ['--import', pathToFileURL(mock).href, join(scripts, 'parity-pin.mjs')], { cwd: resolve(tmpdir()), env: { ...process.env, GITHUB_OUTPUT: output, NODE_OPTIONS: '', NODE_PATH: '' }, encoding: 'utf8', timeout: 10_000 });
  assert.equal(result.status, 0, result.stderr); assert.equal(result.stdout, sha + '\n'); assert.equal(readFileSync(output, 'utf8'), 'sha=' + sha + '\n');
});

test('scheduled conformance uses merged main through the separate drift workflow', () => {
  const pinned = readFileSync(new URL('../workflows/conformance.yml', import.meta.url), 'utf8');
  const drift = readFileSync(new URL('../workflows/parity-drift.yml', import.meta.url), 'utf8');
  assert.doesNotMatch(pinned, /^ {2}(schedule|workflow_dispatch):/m);
  assert.match(pinned, /^ {2}pull_request:/m);
  assert.match(drift, /^ {2}schedule:/m);
  assert.match(drift, /^ {2}workflow_dispatch:/m);
  assert.doesNotMatch(drift, /^ {2}pull_request:/m);
  assert.match(drift, /^ {10}ref: main$/m);
});
