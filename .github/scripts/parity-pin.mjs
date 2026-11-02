import { appendFileSync, lstatSync, readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { fileURLToPath, pathToFileURL, URL } from 'node:url';
import process from 'node:process';
import console from 'node:console';

const pinFile = fileURLToPath(new URL('../../conformance/parity.lock', import.meta.url));

export function readPin(file = pinFile) {
  try {
    const stat = lstatSync(file);
    if (!stat.isFile() || stat.isSymbolicLink() || stat.size < 40 || stat.size > 42) throw new Error();
    const text = readFileSync(file, 'utf8');
    const match = /^([a-fA-F0-9]{40})(?:\r?\n)?$/.exec(text);
    if (!match || match[0] !== text) throw new Error();
    return match[1].toLowerCase();
  } catch {
    throw new Error('conformance/parity.lock must be a regular file containing exactly one 40-hex commit SHA.');
  }
}

export function assertMerged(sha, comparison) {
  if (!['ahead', 'identical'].includes(comparison?.status)
      || comparison?.base_commit?.sha !== sha
      || comparison?.merge_base_commit?.sha !== sha) {
    throw new Error('The pinned parity revision is not merged into prism-parity main; refusing to publish.');
  }
}

function main() {
  const args = process.argv.slice(2);
  if (args.length > 1 || (args.length === 1 && args[0] !== '--release')) throw new Error('Usage: parity-pin.mjs [--release]');
  const sha = readPin();
  if (args[0] === '--release') {
    let comparison;
    try {
      comparison = JSON.parse(execFileSync('gh', ['api', '--method', 'GET', '-H', 'Cache-Control: no-cache', `repos/Particle-Academy/prism-parity/compare/${sha}...main`], { encoding: 'utf8', timeout: 60_000, maxBuffer: 16 * 1024 * 1024, stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true }));
    } catch {
      throw new Error('Could not verify the pinned parity revision against prism-parity main; refusing to publish.');
    }
    assertMerged(sha, comparison);
    console.log(`Pinned corpus ${sha} is merged into prism-parity main.`);
  } else {
    if (process.env.GITHUB_OUTPUT) appendFileSync(process.env.GITHUB_OUTPUT, `sha=${sha}\n`);
    console.log(sha);
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try { main(); } catch (error) { console.error(error.message); process.exitCode = 1; }
}
