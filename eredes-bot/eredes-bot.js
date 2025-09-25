const { chromium } = require('playwright');
const readline = require('readline');
const path = require('path');
const fs = require('fs');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

// Configurações ultra-otimizadas
const CONFIG = {
  timeouts: {
    elementWait: 10000,    // Timeout máximo para encontrar elementos
    navigationWait: 15000,  // Timeout para navegação
    downloadWait: 30000,   // Timeout para download
    minStabilization: 100, // Tempo mínimo para estabilização
    maxStabilization: 1000 // Tempo máximo para estabilização
  },
  retries: 3
};

function askQuestion(question) {
  return new Promise((resolve) => {
    rl.question(question, (answer) => {
      resolve(answer.trim());
    });
  });
}

// Função para esperar elemento estar pronto e clicar imediatamente
async function waitAndClick(page, selectors, description, timeout = CONFIG.timeouts.elementWait) {
  const selectorArray = Array.isArray(selectors) ? selectors : [selectors];
  console.log(`🔍 Procurando: ${description}`);
  
  for (const selector of selectorArray) {
    try {
      const element = page.locator(selector).first();
      
      // Esperar elemento estar visível E clicável
      await element.waitFor({ 
        state: 'visible', 
        timeout: timeout / selectorArray.length 
      });
      
      // Verificar se elemento é clicável (não coberto por outro elemento)
      await element.waitFor({ 
        state: 'attached', 
        timeout: 1000 
      });
      
      // Espera mínima para estabilização
      await page.waitForTimeout(CONFIG.timeouts.minStabilization);
      
      // Tentar clicar
      await element.click({ timeout: 2000 });
      console.log(`✅ ${description} - clicado com sucesso`);
      return true;
      
    } catch (error) {
      console.log(`⏳ Tentativa com "${selector}" falhou, tentando próximo...`);
      continue;
    }
  }
  
  throw new Error(`❌ Não foi possível encontrar/clicar: ${description}`);
}

// Função para preencher campo apenas quando estiver pronto
async function waitAndFill(page, selector, value, description) {
  console.log(`📝 Preenchendo: ${description}`);
  const element = page.locator(selector);
  
  // Esperar campo estar visível e editável
  await element.waitFor({ state: 'visible', timeout: CONFIG.timeouts.elementWait });
  await element.waitFor({ state: 'attached', timeout: 2000 });
  
  // Limpar e preencher
  await element.fill('');
  await element.fill(value);
  console.log(`✅ ${description} preenchido`);
}

// Espera inteligente por mudança de estado/URL
async function waitForStateChange(page, condition, timeout = 10000) {
  const startTime = Date.now();
  const checkInterval = 200; // Verificar a cada 200ms
  
  while (Date.now() - startTime < timeout) {
    try {
      if (await condition()) {
        return true;
      }
    } catch (e) {
      // Continuar verificando
    }
    await page.waitForTimeout(checkInterval);
  }
  return false;
}

// Helper: mapear mês PT (abreviação -> nome completo)
function mapPtMonth(monthInput) {
  const m = (monthInput || '').trim().toLowerCase();
  const table = {
    'jan': 'janeiro',
    'fev': 'fevereiro',
    'mar': 'março',
    'abr': 'abril',
    'mai': 'maio',
    'jun': 'junho',
    'jul': 'julho',
    'ago': 'agosto',
    'set': 'setembro',
    'out': 'outubro',
    'nov': 'novembro',
    'dez': 'dezembro'
  };
  // também aceitar nomes completos
  for (const [abbr, full] of Object.entries(table)) {
    if (m === abbr || m === full) return { abbr, full };
  }
  // fallback: devolver o próprio
  return { abbr: m.slice(0,3), full: m };
}

// Função de debugging para encontrar elementos
async function debugPageElements(page, searchTerm = '') {
  console.log(`🔍 Debug: Analisando elementos da página...`);
  
  try {
    // Pegar todos os botões e links visíveis
    const elements = await page.locator('button, a, input[type="button"], input[type="submit"], [role="button"]').all();
    console.log(`📊 Total de elementos clicáveis encontrados: ${elements.length}`);
    
    const relevantElements = [];
    
    for (let i = 0; i < Math.min(elements.length, 30); i++) {
      try {
        const element = elements[i];
        const isVisible = await element.isVisible();
        
        if (isVisible) {
          const text = await element.innerText({ timeout: 500 }).catch(() => '');
          const tagName = await element.evaluate(el => el.tagName).catch(() => '');
          const className = await element.getAttribute('class').catch(() => '');
          const id = await element.getAttribute('id').catch(() => '');
           
          if (text || className || id) {
            relevantElements.push({
              tag: tagName,
              text: text.trim(),
              class: className,
              id: id
            });
          }
        } 
      } catch (e) {
        // Continuar
      }
    }
    
    // Mostrar elementos relevantes
    console.log('\n📋 Elementos clicáveis encontrados:');
    relevantElements.forEach((el, index) => {
      if (index < 15) { // Mostrar apenas os primeiros 15
        console.log(`${index + 1}. ${el.tag}: "${el.text}" (class: ${el.class || 'none'}, id: ${el.id || 'none'})`);
      }
    });
    
    // Procurar especificamente por termos de exportação
    const exportElements = relevantElements.filter(el => 
      el.text.toLowerCase().includes('export') ||
      el.text.toLowerCase().includes('excel') ||
      el.text.toLowerCase().includes('download') ||
      el.class.toLowerCase().includes('export') ||
      el.class.toLowerCase().includes('excel') ||
      el.id.toLowerCase().includes('export') ||
      el.id.toLowerCase().includes('excel')
    );
    
    if (exportElements.length > 0) {
      console.log('\n🎯 Elementos relacionados com exportação:');
      exportElements.forEach((el, index) => {
        console.log(`${index + 1}. ${el.tag}: "${el.text}" (class: ${el.class}, id: ${el.id})`);
      });
    }
    
    return relevantElements;
    
  } catch (error) {
    console.log(`❌ Debug falhou: ${error.message}`);
    return [];
  }
}

// Verificar e lidar com CAPTCHA de forma otimizada
async function handleCaptchaIfExists(page) {
  // Helper to check visibility safely
  const isVisibleSafe = async (locator, t = 700) => {
    try { return await locator.isVisible({ timeout: t }); } catch { return false; }
  };

  // Common captcha locators
  const recaptchaIframe = page.locator('iframe[title*="reCAPTCHA"], iframe[src*="recaptcha"]');
  const hcaptchaIframe  = page.locator('iframe[src*="hcaptcha"], [class*="hcaptcha"], [id*="hcaptcha"]');
  const challengeIframe = page.locator('iframe[title*="challenge"], iframe[src*="/recaptcha/enterprise/anchor"]:below(:text("Selecionar"))');
  const badge           = page.locator('.grecaptcha-badge'); // v3/invisible badge

  // If nothing captcha-like is visible, skip immediately
  const hasAnyCaptcha = (await isVisibleSafe(recaptchaIframe)) || (await isVisibleSafe(hcaptchaIframe)) || (await isVisibleSafe(badge));
  if (!hasAnyCaptcha) return;

  // If it's just the v3/invisible badge, don't block the flow
  if ((await isVisibleSafe(badge)) && !(await isVisibleSafe(recaptchaIframe)) && !(await isVisibleSafe(hcaptchaIframe))) {
    console.log('ℹ️ CAPTCHA (invisível) detectado — sem interação necessária.');
    return;
  }

  // Give the page a short moment — some captchas auto-pass (score-based)
  await page.waitForTimeout(1500);

  // Try a gentle auto-click on the reCAPTCHA checkbox if present (non-invasive)
  try {
    const frame = page.frameLocator('iframe[title*="reCAPTCHA"]');
    const checkbox = frame.locator('#recaptcha-anchor');
    if (await checkbox.isVisible({ timeout: 800 })) {
      await checkbox.click({ timeout: 1500 });
      await page.waitForTimeout(1200);
    }
  } catch (_) {
    // ignore — fallback below
  }

  // If an actual interactive challenge is shown, then pause for manual resolution.
  const interactive = (await isVisibleSafe(challengeIframe, 800)) || (await isVisibleSafe(page.locator('[class*="rc-anchor-alert"]'), 800));

  if (interactive) {
    console.log('🤖 CAPTCHA interativo detectado — por favor resolva manualmente.');
    await new Promise((resolve) => {
      const tempRl = require('readline').createInterface({ input: process.stdin, output: process.stdout });
      tempRl.question('Pressione ENTER após resolver o CAPTCHA... ', () => { tempRl.close(); resolve(); });
    });
  } else {
    console.log('✅ CAPTCHA detetado mas sem necessidade de interação — a continuar.');
  }
}

// === Helpers para export directo ===
async function getCsrfIfAny(page, context) {
  try {
    const meta = await page.evaluate(() => {
      const m = document.querySelector('meta[name="csrf-token"], meta[name="csrf"]');
      if (m) return m.getAttribute('content');
      if (window.__CSRF__) return window.__CSRF__;
      if (window.__env && window.__env.csrfToken) return window.__env.csrfToken;
      return null;
    });
    if (meta) return meta;
  } catch (_) {}
  try {
    const cookies = await context.cookies();
    const c = cookies.find(c => /csrf|xsrf|token/i.test(c.name));
    if (c) return c.value;
  } catch (_) {}
  return null;
}

function detectExportOnce(page, timeoutMs = 8000) {
  return new Promise((resolve) => {
    let done = false;
    const timer = setTimeout(() => {
      if (!done) { done = true; page.off('request', onReq); resolve(null); }
    }, timeoutMs);
    function onReq(req) {
      try {
        const url = req.url().toLowerCase();
        const hint = url.includes('export') || url.includes('excel') || url.includes('download');
        if (hint) {
          done = true;
          clearTimeout(timer);
          page.off('request', onReq);
          resolve({ url: req.url(), method: req.method(), headers: req.headers(), postData: req.postData() });
        }
      } catch (_) {
        // ignore
      }
    }
    page.on('request', onReq);
  });
}


// Função principal otimizada
async function extractERedesDataUltraFast() {
  let browser = null;
  let page = null;
  const startTime = Date.now();
  
  try {
    console.log('⚡ E-REDES Bot Ultra-Fast v7.0\n');
    
    // Inputs
    const nif = await askQuestion('📋 NIF: ');
    const password = await askQuestion('🔒 Password: ');
    const cpe = await askQuestion('⚡ CPE: ');
    const month = await askQuestion('📅 Mês (mai, jun, jul): ');
    
    rl.close();
    
    const cleanCPE = cpe.replace(/\s/g, '');
    console.log('🚀 Iniciando navegação otimizada...\n');
    
    // Setup downloads
    const downloadPath = path.join(process.cwd(), 'downloads');
    if (!fs.existsSync(downloadPath)) {
      fs.mkdirSync(downloadPath, { recursive: true });
    }
    
    // Browser otimizado
    browser = await chromium.launch({
      headless: false,
      args: [
        '--start-maximized',
        '--disable-web-security',
        '--disable-features=VizDisplayCompositor',
        '--disable-blink-features=AutomationControlled'
      ]
    });
    
    const context = await browser.newContext({
      viewport: null,
      acceptDownloads: true,
      userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    });
    
    // Bloquear recursos pesados
    await context.route('**/*', (route) => {
      const request = route.request();
      const resourceType = request.resourceType();
      
      if (['stylesheet', 'image', 'media', 'font'].includes(resourceType)) {
        route.abort();
      } else {
        route.continue();
      }
    });
    
    page = await context.newPage();
    
    // === ETAPA 1: LOGIN ===
    console.log('📄 Navegando para login...');
    await page.goto('https://balcaodigital.e-redes.pt/login', {
      waitUntil: 'domcontentloaded',
      timeout: CONFIG.timeouts.navigationWait
    });
    
    // Aceitar cookies se aparecer
    try {
      const cookieButton = page.locator('text=Aceitar todos os cookies');
      await cookieButton.waitFor({ state: 'visible', timeout: 2000 });
      await cookieButton.click();
      console.log('🍪 Cookies aceitos');
    } catch (e) {
      console.log('ℹ️ Sem banner de cookies');
    }
    
    // Selecionar Particular
    await waitAndClick(page, [
      'text=Particular',
      'button:has-text("Particular")',
      '[class*="particular"]'
    ], 'Particular');

   

    
    // Preencher credenciais
    await waitAndFill(page, 'input[name="username"]', nif, 'NIF');
    await waitAndFill(page, 'input[name="labelPassword"]', password, 'Password');
    
    // Verificar CAPTCHA
    await handleCaptchaIfExists(page);
    
    // Login
    const currentUrl = page.url();
    await waitAndClick(page, [
      'button:has-text("Entrar")',
      'input[type="submit"]',
      '[class*="login-button"]'
    ], 'Botão de login');
    
    // Esperar navegação pós-login
    console.log('⏳ Aguardando redirecionamento...');
    const loginSuccess = await waitForStateChange(page, async () => {
      return page.url() !== currentUrl && !page.url().includes('/login');
    }, CONFIG.timeouts.navigationWait);
    
    if (!loginSuccess) {
      throw new Error('Login falhou - sem redirecionamento');
    }
    
    console.log('✅ Login bem-sucedido!');

    
    
    // === ETAPA 2: NAVEGAÇÃO ===
    
    // Verificar se já estamos na home, senão navegar
    if (!page.url().includes('/home')) {
      await page.goto('https://balcaodigital.e-redes.pt/home', {
        waitUntil: 'domcontentloaded'
      });
    }
    
    // Os meus locais - seletor mais específico
    await waitAndClick(page, [
      'nz-card:has-text("Os meus locais")',
      'text=Os meus locais',
      'h3:has-text("Os meus locais")',
      '[class*="card"]:has-text("locais")'
    ], 'Os meus locais');
    
    // Consumos e potências
    await waitAndClick(page, [
      'text=Produção, consumos e potências',
      'text=consumos e potências',
      'a:has-text("consumos")'
    ], 'Consumos e potências');
    
    // Consultar histórico - seletor mais específico
    await waitAndClick(page, [
      'div:has-text("Consultar histórico")',
      'text=Consultar histórico',
      'button:has-text("histórico")',
      'a:has-text("histórico")'
    ], 'Consultar histórico');

    try {
      // Pequena espera por qualquer navegação intermédia
      await waitForStateChange(page, async () => page.url().includes('/consumptions'), 5000);
    
      if (!page.url().includes('/consumptions/history')) {
        console.log('↪️ Endpoint incorreto detectado (/consumptions). A navegar para /consumptions/history ...');
        await page.goto('https://balcaodigital.e-redes.pt/consumptions/history', {
          waitUntil: 'domcontentloaded',
          timeout: CONFIG.timeouts.navigationWait
        });
      }
    } catch (e) {
      // Fallback duro: garante que estamos no /history
      if (!page.url().includes('/consumptions/history')) {
        await page.goto('https://balcaodigital.e-redes.pt/consumptions/history', { waitUntil: 'domcontentloaded' });
      }
    }
    
    // Pequena espera para estabilizar antes dos próximos seletores
    await page.waitForTimeout(500);
    
    // === ETAPA 3: SELEÇÕES ===
    
    // Selecionar CPE - usar seletor mais específico
    await waitAndClick(page, [
      `li:has-text("${cleanCPE}")`,
      `[role="listitem"]:has-text("${cleanCPE}")`,
      `text=${cleanCPE}`,
      `div:has-text("${cleanCPE}")`
    ], `CPE ${cleanCPE}`);
    
    // Selecionar mês (robusto: aceita "jun" ou "junho")
    const { abbr: monthAbbr, full: monthFull } = mapPtMonth(month);
    console.log(`📅 Selecionando mês ${month} (→ tenta: ${monthFull}/${monthAbbr})...`);

    // Alguns datepickers usam input role textbox com label/placeholder "Selecionar mês"
    let monthSelector = page.getByRole('textbox', { name: /selecionar mês/i });
    try {
      await monthSelector.waitFor({ state: 'visible', timeout: 4000 });
      await monthSelector.click({ timeout: 2000 });
    } catch (_) {
      // fallback: procurar outros elementos clicáveis que abram o seletor
      monthSelector = page.locator('[placeholder*="Selecionar mês" i], input[aria-label*="Selecionar mês" i], nz-select, [class*="month" i]');
      await monthSelector.first().click({ timeout: 5000 });
    }

    // Dar tempo para o overlay abrir
    await page.waitForTimeout(500);

    // Tentar selecionar no overlay (Angular CDK / nz-select)
    const overlayOptionSelectors = [
      `.cdk-overlay-pane [role="option"]:has-text("${monthFull}")`,
      `.cdk-overlay-pane [role="option"]:has-text("${monthAbbr}")`,
      `.cdk-overlay-pane li:has-text("${monthFull}")`,
      `.cdk-overlay-pane li:has-text("${monthAbbr}")`,
      `li:has-text("${monthFull}")`,
      `li:has-text("${monthAbbr}")`,
      `option:has-text("${monthFull}")`,
      `option:has-text("${monthAbbr}")`,
      `text=${monthFull}`,
      `text=${monthAbbr}`
    ];

    let monthSelected = false;
    for (const sel of overlayOptionSelectors) {
      try {
        const el = page.locator(sel).first();
        await el.waitFor({ state: 'visible', timeout: 600 });
        await el.click({ timeout: 800 });
        monthSelected = true;
        console.log(`✅ Mês selecionado via overlay: ${sel}`);
        break;
      } catch (_) { /* tenta o próximo */ }
    }

    // Se não conseguiu via overlay, tentar escrever no input e Enter
    if (!monthSelected) {
      try {
        const inputEl = page.getByRole('textbox', { name: /selecionar mês/i });
        await inputEl.fill('');
        await inputEl.type(monthFull, { delay: 50 });
        await page.keyboard.press('Enter');
        await page.waitForTimeout(400);
        monthSelected = true;
        console.log(`✅ Mês selecionado por escrita direta: ${monthFull}`);
      } catch (_) {}
    }

    if (!monthSelected) {
      throw new Error(`❌ Não foi possível encontrar/clicar: Mês ${month} (tentado: ${monthFull}/${monthAbbr})`);
    }
    
    // === ETAPA 4: EXPORTAÇÃO ===
    
    console.log('💾 Aguardando dados carregarem...');
    
    // Esperar dados carregarem após seleção do mês
    await page.waitForTimeout(3000);
    
    // Tentar o seletor exato que funciona primeiro
    console.log('🔍 Procurando "Exportar excel"...');
    
    const exportSelectors = [
      'a:has-text("Exportar excel")',  // Seletor que funciona no teste
      'locator("a").filter({ hasText: "Exportar excel" })',
      'text=Exportar excel',
      'button:has-text("Exportar excel")',
      'text=Exportar Excel',
      'button:has-text("Exportar Excel")',
      'link:has-text("Exportar excel")',
      'a[href*="export"]',
      'a:has-text("excel")'
    ];
    
    let exportClicked = false;
    
    // Tentar os seletores principais primeiro
    for (const selector of exportSelectors) {
      try {
        console.log(`🔍 Tentando: ${selector}`);
        const element = page.locator(selector).first();
        
        // Esperar elemento estar visível
        await element.waitFor({ state: 'visible', timeout: 5000 });
        
        // Verificar se está realmente visível
        const isVisible = await element.isVisible();
        if (isVisible) {
          await element.click({ timeout: 3000 });
          console.log(`✅ "Exportar excel" encontrado e clicado!`);
          exportClicked = true;
          break;
        }
        
      } catch (e) {
        console.log(`⏳ ${selector} - não encontrado: ${e.message.split('\n')[0]}`);
        continue;
      }
    }
    
    // Se ainda não encontrou, fazer busca mais ampla
    if (!exportClicked) {
      console.log('🔍 Busca ampliada por elementos de exportação...');
      
      // Procurar por qualquer link (a) que contenha "excel" ou "export"
      const allLinks = await page.locator('a').all();
      console.log(`📊 Verificando ${allLinks.length} links...`);
      
      for (const link of allLinks) {
        try {
          const isVisible = await link.isVisible();
          if (isVisible) {
            const text = await link.innerText({ timeout: 500 }).catch(() => '');
            const href = await link.getAttribute('href').catch(() => '');
            
            if (text.toLowerCase().includes('excel') || 
                text.toLowerCase().includes('export') ||
                href.includes('export')) {
              
              console.log(`🎯 Link encontrado: "${text}" (href: ${href})`);
              await link.click();
              exportClicked = true;
              break;
            }
          }
        } catch (e) {
          continue;
        }
      }
    }
    
    if (!exportClicked) {
      throw new Error('❌ Link "Exportar excel" não encontrado');
    }
    
    console.log('⏳ Aguardando modal de exportação...');
    await page.waitForTimeout(2000);
    
    // Confirmar exportação
    console.log('⬇️ Preparar captura do endpoint de export...');
    const capturePromise = detectExportOnce(page, 8000);

    await waitAndClick(page, [
      'button:has-text("Exportar")',
      'input[value="Exportar"]',
      '[class*="export-button"]'
    ], 'Confirmar Exportação');

    const captured = await capturePromise;
    if (!captured) {
      throw new Error('Não foi possível capturar a URL de export (timeout).');
    }
    console.log('🎯 Export capturado:', captured.url, 'method:', captured.method);

    // Cabeçalhos mínimos + CSRF
    const headers = {
      'Referer': page.url(),
      'Origin': new URL(page.url()).origin,
      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
    };
    const csrf = await getCsrfIfAny(page, context);
    if (csrf) headers['X-CSRF-Token'] = csrf;

    // Fazer a chamada directa com o contexto autenticado
    let resp;
    if ((captured.method || 'GET').toUpperCase() === 'POST') {
      const body = captured.postData || undefined;
      const extra = body ? { data: body } : {};
      resp = await context.request.post(captured.url, { headers, timeout: CONFIG.timeouts.downloadWait, ...extra });
    } else {
      resp = await context.request.get(captured.url, { headers, timeout: CONFIG.timeouts.downloadWait });
    }

    if (!resp.ok()) {
      const snippet = (await resp.text().catch(() => ''))?.slice(0, 200);
      throw new Error(`Pedido directo falhou (${resp.status()}). Resumo: ${snippet}`);
    }

    const buffer = await resp.body();
    const timestamp = new Date().toISOString().split('T')[0];
    const fileName = `eredes-${nif}-${month}-${timestamp}.xlsx`;
    const filePath = path.join(downloadPath, fileName);
    fs.writeFileSync(filePath, buffer);
    
    // === RESULTADO ===
    const endTime = Date.now();
    const duration = ((endTime - startTime) / 1000).toFixed(1);
    
    console.log(`\n🎉 SUCESSO TOTAL!`);
    console.log(`⚡ Tempo recorde: ${duration}s`);
    console.log(`📁 Arquivo: ${fileName}`);
    console.log(`📍 Local: ${downloadPath}`);
    
    if (fs.existsSync(filePath)) {
      const stats = fs.statSync(filePath);
      console.log(`📊 Tamanho: ${(stats.size / 1024).toFixed(1)} KB`);
    }
    
  } catch (error) {
    console.error(`\n❌ ERRO: ${error.message}`);
    
    if (page) {
      console.log(`📍 URL atual: ${page.url()}`);
      
      // Screenshot de erro
      try {
        const errorScreenshot = path.join(process.cwd(), `erro-debug-${Date.now()}.png`);
        await page.screenshot({ path: errorScreenshot, fullPage: true });
        console.log(`📸 Debug screenshot: ${errorScreenshot}`);
      } catch (e) {}
    }
    
    console.log('\n🔧 Soluções:');
    console.log('• Verificar credenciais');
    console.log('• Verificar CPE e mês');
    console.log('• Tentar novamente (site pode estar lento)');
    
  } finally {
    if (browser) {
      // Pequena pausa para ver resultado
      await new Promise(resolve => setTimeout(resolve, 3000));
      await browser.close();
    }
    console.log('\n👋 Bot finalizado.');
    process.exit(0);
  }
}

// Iniciar
console.log('⚡ E-REDES BOT ULTRA-FAST v7.0');
console.log('==============================');
console.log('🎯 Esperas mínimas necessárias');
console.log('🚀 Máxima velocidade de execução');
console.log('⚡ Detecção inteligente de estados\n');

extractERedesDataUltraFast();