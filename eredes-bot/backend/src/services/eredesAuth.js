const { chromium } = require('playwright');

const LOGIN_URL = 'https://balcaodigital.e-redes.pt/login';
const AFTER_LOGIN_URL = 'https://balcaodigital.e-redes.pt/consumptions/history';

class InvalidCredentialsError extends Error {
  constructor(message = 'Credenciais inválidas. Verifique o NIF e a password.') {
    super(message);
    this.name = 'InvalidCredentialsError';
  }
}

class CaptchaError extends Error {
  constructor(message = 'Foi detetado um desafio de segurança. Por favor tente novamente mais tarde.') {
    super(message);
    this.name = 'CaptchaError';
  }
}

async function fillInput(page, selectors, value) {
  for (const selector of selectors) {
    const locator = page.locator(selector).first();
    if (await locator.count()) {
      try {
        await locator.fill('');
        await locator.type(value, { delay: 50 });
        return true;
      } catch (error) {
        continue;
      }
    }
  }
  return false;
}

function createWaitForSuccess(page) {
  const successRegex = /balcaodigital\.e-redes\.pt\/(home|dashboard|consumptions|contracts|profile)/i;
  return page.waitForURL((url) => successRegex.test(url.toString()), {
    timeout: 60_000,
    waitUntil: 'load',
  });
}

async function waitForLoginOutcome(page) {
  const successPromise = createWaitForSuccess(page).then(() => 'success').catch(() => null);
  const errorPromise = page
    .waitForSelector('.ant-form-item-explain-error, .ant-alert-error', { timeout: 60_000 })
    .then(async (element) => ({
      type: 'error',
      message: (await element.innerText()).trim(),
    }))
    .catch(() => null);

  const captchaPromise = page
    .waitForSelector('iframe[src*="recaptcha"], .grecaptcha-badge', { timeout: 60_000 })
    .then(() => 'captcha')
    .catch(() => null);

  const outcome = await Promise.race([successPromise, errorPromise, captchaPromise].filter(Boolean));

  if (!outcome) {
    return 'timeout';
  }

  if (outcome === 'success') {
    return 'success';
  }

  if (outcome === 'captcha') {
    throw new CaptchaError();
  }

  if (typeof outcome === 'object' && outcome.type === 'error') {
    throw new InvalidCredentialsError(outcome.message || undefined);
  }

  return outcome;
}

async function loginToERedes({ nif, password }) {
  const browser = await chromium.launch({ headless: process.env.PLAYWRIGHT_HEADLESS !== 'false' });
  const context = await browser.newContext();
  const page = await context.newPage();

  let authHeader = null;

  page.on('request', (request) => {
    const headers = request.headers();
    if (!authHeader && headers['authorization-request']) {
      authHeader = headers['authorization-request'];
    }
  });

  try {
    await page.goto(LOGIN_URL, { waitUntil: 'networkidle' });

    const filledNif = await fillInput(page, ['input#username', 'input[name="username"]', 'input[type="text"]'], nif);
    const filledPassword = await fillInput(page, ['input#labelPassword', 'input[name="password"]', 'input[type="password"]'], password);

    if (!filledNif || !filledPassword) {
      throw new Error('Não foi possível preencher o formulário de login.');
    }

    const submitButton = page.locator('button[type="submit"], button.ant-btn-primary');
    if (await submitButton.count()) {
      await submitButton.first().click();
    } else {
      await page.keyboard.press('Enter');
    }

    const outcome = await waitForLoginOutcome(page);

    if (outcome === 'timeout') {
      const timeoutError = new Error('Timeout ao aguardar confirmação do login.');
      timeoutError.status = 504;
      throw timeoutError;
    }

    if (outcome !== 'success') {
      const genericError = new Error('Não foi possível confirmar o estado do login.');
      genericError.status = 500;
      throw genericError;
    }

    for (let attempt = 0; attempt < 4 && !authHeader; attempt += 1) {
      await page.goto(AFTER_LOGIN_URL, { waitUntil: 'networkidle' });
      await page.waitForTimeout(1000);
    }

    if (!authHeader) {
      throw new Error('Não foi possível obter o token de autorização.');
    }

    const cookies = await context.cookies();
    const cookieHeader = cookies.map((cookie) => `${cookie.name}=${cookie.value}`).join('; ');

    return {
      authToken: authHeader,
      cookieHeader,
    };
  } finally {
    await browser.close();
  }
}

module.exports = {
  loginToERedes,
  InvalidCredentialsError,
  CaptchaError,
};
