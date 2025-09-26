/**
 * e-redes-auto.js — direct API version with optional automated login via Playwright
 *
 * Adds an `--auto-auth` flow that opens a real browser (Playwright), performs
 * login (automatically if env vars provided, otherwise manual) and extracts the
 * `authorization-request` header plus session cookies so the script can call
 * the EDM endpoint without the developer manually copying tokens.
 *
 * IMPORTANT
 *  - This still does NOT bypass/solve reCAPTCHA. If a reCAPTCHA appears the
 *    script will pause and wait for you to complete the login in the opened
 *    browser window (or provide credentials via env vars if no CAPTCHA).
 *  - Requires installing Playwright: `npm i -D playwright` (or `playwright-core`
 *    plus a browser). Node >= 18.
 *
 * ENV variables used for automatic login (optional):
 *  - EREDES_NIF       -> NIF (or username) for login
 *  - EREDES_PASSWORD  -> password
 *  - EREDES_AUTH_REQ  -> (optional) if already have token, skip auto-auth
 *  - EREDES_COOKIE    -> (optional) raw Cookie header to skip auto-auth
 *
 * USAGE
 *  node e-redes-auto.js --auto-auth true --cpe <CPE> --start "..." --end "..."
 *  node e-redes-auto.js --auto-auth true --print-auth true   # abre browser, faz login e imprime token+cookies
 */


const readline = require('readline');
const fs = require('fs');
const path = require('path');
function ask(question) {
  return new Promise((resolve) => {
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
    rl.question(question, (ans) => { rl.close(); resolve(ans.trim()); });
  });
}

const DEFAULT_URL = 'https://balcaodigital.e-redes.pt/ms/reading/data-usage/edm/get';

const WAIT_AUTH_TIMEOUT_MS = 60_000; // time to wait for token when using browser

function parseArgs(argv) {
  const out = {};
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith('--')) {
      const key = a.slice(2);
      const next = argv[i + 1];
      if (next && !next.startsWith('--')) {
        out[key] = next;
        i++;
      } else {
        out[key] = true;
      }
    }
  }
  return out;
}

function buildCookieHeaderFromEnv() {
  if (process.env.EREDES_COOKIE && process.env.EREDES_COOKIE.trim() !== '') {
    return process.env.EREDES_COOKIE.trim();
  }
  const parts = [];
  const pushIf = (name) => {
    const v = process.env[name];
    if (v && v.trim() !== '') parts.push(`${name.replace('EREDES_', '')}=${v.trim()}`);
  };
  pushIf('EREDES_PHPSESSID');
  pushIf('EREDES_SIMPLESAML');
  pushIf('EREDES_AAT');
  pushIf('EREDES_OTHER');
  return parts.join('; ');
}

function safeJson(text) {
  try { return JSON.parse(text); } catch { return null; }
}

function ensureArray(x) {
  if (x == null) return [];
  return Array.isArray(x) ? x : [x];
}

/**
 * Flattens the EDM response JSON to a table of rows:
 * Columns: cpe, contractId, measuringPointId, loadCurveTimestamp, meterLoadCurve, meterLoadCurveStatus, meterLoadCurveUnitMeasurement
 */
function flattenEdmToRows(json) {
  const rows = [];
  if (!json) return rows;

  // Prefer the EDM shape used by E-REDES: Body.Result.utilitiesDevices
  try {
    const devices = (((json.Body || {}).Result || {}).utilitiesDevices) || [];
    for (const dev of devices) {
      const measuringPointId = dev.meterReaderSerialNumber || dev.measuringPointId || dev.id || '';
      const meterLoadCurves = ensureArray(dev.meterLoadCurves);
      for (const mc of meterLoadCurves) {
        const register = mc.register || '';
        const loadCurves = ensureArray(mc.loadCurves);
        for (const e of loadCurves) {
          const ts = e.loadCurveTimestamp || e.timestamp || e.date || e.datetime || e.time || null;
          const val = e.meterLoadCurve ?? e.value ?? e.kwh ?? e.energy ?? null;
          const status = e.meterLoadCurveStatus ?? e.status ?? null;
          const unit = e.meterLoadCurveUnitMeasurement ?? e.unit ?? 'kwh';
          if (ts != null && val != null) {
            rows.push({
              cpe: dev.cpe || dev.CPE || '',
              contractId: '',
              measuringPointId,
              register,
              loadCurveTimestamp: ts,
              meterLoadCurve: val,
              meterLoadCurveStatus: status,
              meterLoadCurveUnitMeasurement: unit,
            });
          }
        }
      }
    }
    if (rows.length > 0) return rows;
  } catch (_) {}

  // Fallbacks for alternative shapes
  const accounts = ensureArray(json.accounts || json.data || json.result || json);
  for (const acc of accounts) {
    const contracts = ensureArray(acc.contracts || acc.contract || acc.children);
    for (const c of contracts) {
      const measuringPoints = ensureArray(c.measuringPoints || c.measuring_point || c.points);
      for (const mp of measuringPoints) {
        const cpe = (mp.cpe || mp.CPE || mp.id || mp.identifier || c.cpe || acc.cpe) || '';
        const contractId = c.contractId || c.id || '';
        const measuringPointId = mp.measuringPointId || mp.id || '';

        const loadCurves = ensureArray(mp.loadCurves || mp.load_curve || mp.curves || mp.entries);
        for (const lc of loadCurves) {
          const entries = ensureArray(lc.entries || lc.loadCurves || lc.load_curve || lc);
          for (const e of entries) {
            const ts = e.loadCurveTimestamp || e.timestamp || e.date || e.datetime || e.time || null;
            const val = e.meterLoadCurve ?? e.value ?? e.kwh ?? e.energy ?? null;
            const status = e.meterLoadCurveStatus ?? e.status ?? null;
            const unit = e.meterLoadCurveUnitMeasurement ?? e.unit ?? 'kwh';
            if (ts != null && val != null) {
              rows.push({
                cpe,
                contractId,
                measuringPointId,
                register: e.register || '',
                loadCurveTimestamp: ts,
                meterLoadCurve: val,
                meterLoadCurveStatus: status,
                meterLoadCurveUnitMeasurement: unit,
              });
            }
          }
        }
      }
    }
  }
  return rows;
}

function writeTableToCSV(rows, filePath) {
  const headers = [
    'cpe',
    'contractId',
    'measuringPointId',
    'loadCurveTimestamp',
    'meterLoadCurve',
    'meterLoadCurveStatus',
    'meterLoadCurveUnitMeasurement'
  ];
  const esc = (v) => {
    if (v == null) return '';
    const s = String(v);
    if (/[",\n]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
    return s;
  };
  const lines = [headers.join(',')];
  for (const r of rows) {
    lines.push(headers.map(h => esc(r[h])).join(','));
  }
  fs.writeFileSync(filePath, lines.join('\n'));
}

function writeTableToXLSX(rows, filePath) {
  let XLSX;
  try {
    XLSX = require('xlsx');
  } catch (e) {
    // Fallback: write CSV with same basename if xlsx module is missing
    const csvPath = filePath.replace(/\.xlsx$/i, '.csv');
    writeTableToCSV(rows, csvPath);
    console.warn(`[excel] Package "xlsx" not installed. Wrote CSV instead: ${csvPath}`);
    return { path: csvPath, format: 'csv' };
  }
  const headers = [
    'cpe',
    'contractId',
    'measuringPointId',
    'loadCurveTimestamp',
    'meterLoadCurve',
    'meterLoadCurveStatus',
    'meterLoadCurveUnitMeasurement'
  ];
  const aoa = [headers];
  for (const r of rows) {
    aoa.push(headers.map(h => r[h] ?? ''));
  }
  const ws = XLSX.utils.aoa_to_sheet(aoa);
  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, ws, 'EDM');
  XLSX.writeFile(wb, filePath);
  return { path: filePath, format: 'xlsx' };
}

function formatPtDate(date, useUTC = false) {
  const y = useUTC ? date.getUTCFullYear() : date.getFullYear();
  const m = String((useUTC ? date.getUTCMonth() : date.getMonth()) + 1).padStart(2, '0');
  const d = String(useUTC ? date.getUTCDate() : date.getDate()).padStart(2, '0');
  return `${y}/${m}/${d}`;
}
function formatPtTime(date, useUTC = false) {
  const hh = String(useUTC ? date.getUTCHours() : date.getHours()).padStart(2, '0');
  const mm = String(useUTC ? date.getUTCMinutes() : date.getMinutes()).padStart(2, '0');
  return `${hh}:${mm}`;
}
/**
 * rowsToPowerTable
 * @param {Array} rows - flattened EDM rows (kWh per 15 min)
 * @param {Object} opts - { tz: 'local'|'utc', label: 'start'|'end' }
 */
function rowsToPowerTable(rows, opts = {}) {
  const tz = (opts.tz || 'local').toLowerCase();     // 'local' | 'utc'
  const label = (opts.label || 'start').toLowerCase(); // 'start' | 'end'
  const useUTC = tz === 'utc';
  const addMinutes = (d, min) => { const t = new Date(d.getTime()); t.setMinutes(t.getMinutes() + min); return t; };

  const out = [];
  for (const r of rows) {
    if (!r.loadCurveTimestamp || r.meterLoadCurve == null) continue;
    const dt = new Date(r.loadCurveTimestamp); // parse ISO8601 with 'Z'
    const labelTime = label === 'end' ? addMinutes(dt, 15) : dt; // end-of-interval labeling if requested
    
    const powerKw = Number(r.meterLoadCurve) * 4; // kW = kWh * 4
    out.push({
      'Data': formatPtDate(labelTime, useUTC),
      'Hora': formatPtTime(labelTime, useUTC),
      'Consumo registado (kW)': Number.isFinite(powerKw) ? Number(powerKw.toFixed(3)) : '',
    });
  }
  // ordenar por Data+Hora
  out.sort((a, b) => (a.Data + ' ' + a.Hora).localeCompare(b.Data + ' ' + b.Hora));
  return out;
}
  function writePowerCSV(powerRows, filePath) {
    const headers = ['Data', 'Hora', 'Consumo registado (kW)'];
    const esc = (v) => {
      if (v == null) return '';
      const s = String(v);
      if (/[",\n]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
      return s;
    };
    const lines = [headers.join(',')];
    for (const r of powerRows) {
      lines.push(headers.map(h => esc(r[h])).join(','));
    }
    fs.writeFileSync(filePath, lines.join('\n'));
  }
  function writePowerXLSX(powerRows, filePath, alsoRawRows = null) {
    let XLSX;
    try {
      XLSX = require('xlsx');
    } catch (e) {
      const csvPath = filePath.replace(/\.xlsx$/i, '.csv');
      writePowerCSV(powerRows, csvPath);
      console.warn(`[excel] Package "xlsx" not installed. Wrote CSV instead: ${csvPath}`);
      return { path: csvPath, format: 'csv' };
    }
    const headers = ['Data', 'Hora', 'Consumo registado (kW)'];
    const aoa = [headers];
    for (const r of powerRows) {
      aoa.push([r['Data'], r['Hora'], r['Consumo registado (kW)']]);
    }
    const ws1 = XLSX.utils.aoa_to_sheet(aoa);
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws1, 'Potência_15min');
  
    // Opcional: segunda folha com os dados "Raw" achatados
    if (Array.isArray(alsoRawRows) && alsoRawRows.length) {
      const rawHeaders = [
        'cpe','contractId','measuringPointId','register',
        'loadCurveTimestamp','meterLoadCurve','meterLoadCurveStatus','meterLoadCurveUnitMeasurement'
      ];
      const aoaRaw = [rawHeaders];
      for (const r of alsoRawRows) {
        aoaRaw.push(rawHeaders.map(h => r[h] ?? ''));
      }
      const ws2 = XLSX.utils.aoa_to_sheet(aoaRaw);
      XLSX.utils.book_append_sheet(wb, ws2, 'Raw');
    }
    XLSX.writeFile(wb, filePath);
    return { path: filePath, format: 'xlsx' };
  }

async function requestEdm({
  url = DEFAULT_URL,
  cpe,
  start,
  end,
  type = '3',
  formatted = 'false',
  wait = 'true',
  nif_requester = null,
  serial_number = '',
  nif = null,
  authHeader,
  cookieHeader,
  retries = 1,
  timeoutMs = 30000,
}) {
  if (!authHeader || authHeader.trim() === '') {
    throw new Error('Missing authorization-request header. Provide EREDES_AUTH_REQ or use --auto-auth.');
  }
  if (!cpe || !start || !end) {
    throw new Error('Missing required args: --cpe, --start, --end');
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  const body = {
    cpe,
    request_type: String(type),
    start_date: start,
    end_date: end,
    wait: String(wait) === 'true',
    formatted: String(formatted) === 'true',
    nif_requester,
    serial_number,
    nif,
  };

  const headers = {
    'content-type': 'application/json',
    'accept': 'application/json, text/plain, */*',
    'authorization-request': authHeader,
    'Origin': 'https://balcaodigital.e-redes.pt',
    'Referer': 'https://balcaodigital.e-redes.pt/consumptions/history',
    'User-Agent': 'Mozilla/5.0',
  };
  if (cookieHeader) headers['Cookie'] = cookieHeader;

  let lastErr;
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const res = await fetch(url, {
        method: 'POST',
        headers,
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      if (!res.ok) {
        const text = await res.text();
        const json = safeJson(text);
        const msg = json?.message || json?.error || text?.slice(0, 500);
        const err = new Error(`${res.status} ${res.statusText}${msg ? ` — ${msg}` : ''}`);
        err.status = res.status;
        err.payload = json || text;
        throw err;
      }
      clearTimeout(timer);
      return await res.json();
    } catch (e) {
      lastErr = e;
      const s = e.status;
      const transient = e.name === 'AbortError' || e.code === 'ECONNRESET' || e.code === 'ETIMEDOUT';
      const canRetry = transient || s === 401 || s === 403 || s === 429 || s === 502 || s === 503 || s === 504;
      if (attempt < retries && canRetry) {
        await new Promise(r => setTimeout(r, 500 + attempt * 500));
        continue;
      }
      break;
    } finally {
      clearTimeout(timer);
    }
  }
  throw lastErr;
}

// --- Playwright automated login + extraction ---
async function autoAuthWithPlaywright({
  headless = false,
  loginUrl = 'https://balcaodigital.e-redes.pt/login',
  waitTimeout = WAIT_AUTH_TIMEOUT_MS,
}) {
  // dynamic import so script can still run if playwright isn't installed and auto-auth isn't used
  let playwright;
  try {
    playwright = require('playwright');
  } catch (e) {
    throw new Error('Playwright not installed. Run `npm i -D playwright` (and optionally `npx playwright install`).');
  }

  const browser = await playwright.chromium.launch({ headless });
  const context = await browser.newContext();
  const page = await context.newPage();

  let capturedAuth = null;
  // Capture outgoing requests and look for a header called 'authorization-request'
  page.on('request', (req) => {
    try {
      const headers = req.headers();
      if (!capturedAuth && headers['authorization-request']) {
        capturedAuth = headers['authorization-request'];
      }
    } catch (err) { /* ignore */ }
  });

  // Navigate to the default login page
  await page.goto(loginUrl, { waitUntil: 'networkidle' });

  // Attempt automated fill if env vars present
  const nif = process.env.EREDES_NIF;
  const pass = process.env.EREDES_PASSWORD;
  if (nif && pass) {
    // Try a few common selectors; these may need adjusting for site changes.
    const candidates = [
      'input[name="nif"]',
      'input[name="username"]',
      'input[type="text"]',
      'input[id*="nif"]',
    ];
    let filled = false;
    for (const sel of candidates) {
      try {
        const el = await page.$(sel);
        if (el) {
          await el.fill(nif);
          filled = true;
          break;
        }
      } catch (e) {}
    }
    // password
    const passCandidates = ['input[type="password"]', 'input[name="password"]', 'input[id*="pwd"]'];
    for (const sel of passCandidates) {
      try {
        const el = await page.$(sel);
        if (el) {
          await el.fill(pass);
          break;
        }
      } catch (e) {}
    }

    // Try to click a login button
    const btnCandidates = ['button[type="submit"]', 'button[id*="login"]', 'button[class*="login"]'];
    for (const sel of btnCandidates) {
      try {
        const b = await page.$(sel);
        if (b) { await b.click(); break; }
      } catch (e) {}
    }
  } else {
    // No creds provided: ask the user to login manually in the opened browser
    console.log('[auto-auth] No EREDES_NIF / EREDES_PASSWORD env vars found.');
    console.log('[auto-auth] A browser window has been opened. Please login manually and complete any CAPTCHA if present.');
    console.log('[auto-auth] After completing login, press ENTER in this terminal to continue.');

    // Keep the page open and wait for user to press enter
    await waitForEnter();
  }

  // Wait until we capture the header or timeout
  const start = Date.now();
  while (!capturedAuth && (Date.now() - start) < waitTimeout) {
    // We try to stimulate network activity by navigating to the consumption history page
    try {
      await page.goto('https://balcaodigital.e-redes.pt/consumptions/history', { waitUntil: 'networkidle' });
    } catch (e) {}
    // Wait a bit
    await sleep(500);
  }

  if (!capturedAuth) {
    // As fallback, try to read localStorage or cookies where frontend may store token
    try {
      const maybe = await page.evaluate(() => {
        try {
          return {
            local: JSON.stringify(window.localStorage),
            session: JSON.stringify(window.sessionStorage),
          };
        } catch (e) { return null; }
      });
      if (maybe && maybe.local) {
        // heuristic: search values for "authorization" or long strings
        const text = maybe.local + '\n' + maybe.session;
        const m = text.match(/[A-Za-z0-9\-_.]{20,}/g);
        if (m && m.length) capturedAuth = m[0];
      }
    } catch (e) { /* ignore */ }
  }

  // Grab cookies and build Cookie header
  const cookies = await context.cookies();
  const cookieHeader = cookies.map(c => `${c.name}=${c.value}`).join('; ');

  await browser.close();

  if (!capturedAuth) {
    throw new Error('Could not extract authorization-request token. Either CAPTCHA blocked automated flows or site layout changed.');
  }

  return { authHeader: capturedAuth, cookieHeader };
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

function waitForEnter() {
  return new Promise((resolve) => {
    const rl = require('readline').createInterface({ input: process.stdin, output: process.stdout });
    rl.question('', () => {
      rl.close();
      resolve();
    });
  });
}

// --- Main CLI ---
async function main() {
  const args = parseArgs(process.argv);
  let {
    cpe,
    start,
    end,
    type = '3',
    formatted = 'false',
    wait = 'true',
    url = DEFAULT_URL,
    retries = '1',
    'auto-auth': autoAuthFlag,
    headless = 'false',
    'print-auth': printAuthOnly,
    out: outPath,
    tz = 'utc',          // default UTC so 00:00Z -> 00:00 in Excel
    label = 'start',     // 'start' or 'end'
  } = args;

  // Default auto-auth ON when no token is present and user didn't explicitly disable it
  if ((autoAuthFlag === undefined || autoAuthFlag === true || autoAuthFlag === 'true') && !process.env.EREDES_AUTH_REQ) {
    autoAuthFlag = 'true';
  }

  let authHeader = process.env.EREDES_AUTH_REQ || '';
  let cookieHeader = buildCookieHeaderFromEnv();

  try {
    // If we intend to call the API and required args are missing, ask interactively
    if (!printAuthOnly) {
      if (!cpe) cpe = (process.env.EREDES_CPE || await ask('CPE: ')).replace(/\s+/g, '');
      if (!start) start = process.env.EREDES_START || await ask('Start date (YYYY-MM-DD HH:mm:ss): ');
      if (!end) end = process.env.EREDES_END || await ask('End date   (YYYY-MM-DD HH:mm:ss): ');
    }

    if ((!authHeader || authHeader.trim() === '') && (autoAuthFlag === 'true' || autoAuthFlag === true)) {
      console.log('[auto-auth] Starting Playwright flow to obtain token and cookies...');
      const res = await autoAuthWithPlaywright({ headless: headless === 'true' });
      authHeader = res.authHeader;
      cookieHeader = res.cookieHeader;
      console.log('[auto-auth] Obtained token and cookies. Proceeding with API request.');
      if (printAuthOnly === 'true' || printAuthOnly === true) {
        console.log(JSON.stringify({ authHeader, cookieHeader }, null, 2));
        return; // end program after printing
      }
    }

    // If still missing token but env provided, use that
    if (!authHeader || authHeader.trim() === '') {
      if (process.env.EREDES_AUTH_REQ) authHeader = process.env.EREDES_AUTH_REQ;
    }

    if (printAuthOnly === 'true' || printAuthOnly === true) {
      console.log(JSON.stringify({ authHeader, cookieHeader }, null, 2));
      return;
    }

    const data = await requestEdm({
      url,
      cpe,
      start,
      end,
      type,
      formatted,
      wait,
      authHeader,
      cookieHeader,
      retries: Number(retries) || 0,
    });

    // Prepare output file if requested
    const rows = flattenEdmToRows(data);
    if (rows.length === 0) {
        if (outPath) {
          const outJson = path.extname(outPath).toLowerCase() ? outPath : `${outPath}.json`;
          fs.writeFileSync(outJson, JSON.stringify(data, null, 2));
          console.log(`[save] No load curve rows found. Wrote raw JSON to: ${outJson}`);
        }
      } else {
        const powerRows = rowsToPowerTable(rows, { tz, label });
        let target = outPath;
        if (!target || target.trim() === '') {
          const norm = (s) => (s || '').replace(/[-:\s]/g, '').slice(0, 12);
          const cpePart = (cpe || '').replace(/\s+/g, '');
          const startPart = norm(start);
          const endPart = norm(end);
          target = path.join(process.cwd(), `e-redes-${cpePart}-${startPart}-${endPart}.xlsx`);
        }
        const ext = path.extname(target).toLowerCase();
        if (ext === '.xlsx') {
          const info = writePowerXLSX(powerRows, target, rows);
          console.log(`[save] Wrote ${info.format.toUpperCase()} file: ${info.path}`);
        } else if (ext === '.csv') {
          writePowerCSV(powerRows, target);
          console.log(`[save] Wrote CSV file: ${target}`);
        } else if (ext === '.json' || ext === '') {
          fs.writeFileSync(ext === '.json' ? target : `${target}.json`, JSON.stringify(data, null, 2));
          const csvPath = (ext === '.json') ? target.replace(/\.json$/i, '.csv') : `${target}.csv`;
          writePowerCSV(powerRows, csvPath);
          console.log(`[save] Wrote JSON and CSV files: ${ext === '.json' ? target : target + '.json'} , ${csvPath}`);
        } else {
          const fallback = `${target}.xlsx`;
          const info = writePowerXLSX(powerRows, fallback, rows);
          console.log(`[save] Wrote ${info.format.toUpperCase()} file: ${info.path}`);
        }
      }
    console.log(JSON.stringify(data, null, 2));
  } catch (err) {
    console.error('[E-REDES] Request failed:', err?.message || err);
    if (err?.status) console.error('HTTP status:', err.status);
    if (err?.payload) {
      try {
        console.error('Payload:', typeof err.payload === 'string' ? err.payload : JSON.stringify(err.payload));
      } catch {}
    }
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}

module.exports = { requestEdm };