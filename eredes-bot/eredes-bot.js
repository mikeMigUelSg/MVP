const { chromium } = require('playwright');
const readline = require('readline');
const path = require('path');
const fs = require('fs');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

function askQuestion(question) {
  return new Promise((resolve) => {
    rl.question(question, (answer) => {
      resolve(answer.trim());
    });
  });
}

async function waitForCaptcha(page) {
  // Verificar se existe CAPTCHA na página
  const captchaSelectors = [
    'iframe[src*="recaptcha"]',
    '[class*="captcha"]',
    '[id*="captcha"]',
    'canvas',
    '[class*="hcaptcha"]'
  ];
  
  for (const selector of captchaSelectors) {
    const captcha = await page.locator(selector).first();
    const isVisible = await captcha.isVisible().catch(() => false);
    if (isVisible) {
      console.log('🤖 CAPTCHA detectado!');
      return new Promise((resolve) => {
        console.log('⚠️ Resolva o CAPTCHA na janela do browser');
        console.log('⏳ Aguardando resolução automática...');
        rl.question('Se o CAPTCHA não resolver automaticamente, pressione ENTER após resolvê-lo: ', () => {
          resolve();
        });
      });
    }
  }
  
  // Não há CAPTCHA, continuar automaticamente
  return Promise.resolve();
}

async function checkLoginSuccess(page) {
  // Aguardar um pouco para o redirect
  await page.waitForTimeout(3000);
  
  const currentUrl = page.url();
  console.log(`🔍 URL atual: ${currentUrl}`);
  
  // Várias formas de detectar sucesso
  if (currentUrl.includes('/home') || 
      currentUrl.includes('/dashboard') || 
      !currentUrl.includes('/login')) {
    return true;
  }
  
  // Verificar se ainda está na página de login
  const loginElements = [
    'input[name="username"]',
    'input[name="labelPassword"]',
    'text=Entrar'
  ];
  
  for (const selector of loginElements) {
    const element = await page.locator(selector).first();
    const isVisible = await element.isVisible().catch(() => false);
    if (isVisible) {
      return false; // Ainda na página de login
    }
  }
  
  return true; // Assumir sucesso se não encontrar elementos de login
}

async function extractERedesDataAuto() {
  let browser = null;
  
  try {
    console.log('🚀 E-REDES Bot - Totalmente Automático\n');
    
    // Input do utilizador
    const nif = await askQuestion('📋 NIF: ');
    const password = await askQuestion('🔒 Password: ');
    const cpe = await askQuestion('⚡ CPE: ');
    const month = await askQuestion('📅 Mês: ');
    
    const cleanCPE = cpe.replace(/\s/g, '');
    console.log(`✅ Dados recebidos - CPE: ${cleanCPE}, Mês: ${month}\n`);
    
    // Fechar readline input após receber dados
    rl.removeAllListeners();
    
    // Criar pasta downloads
    const downloadPath = path.join(process.cwd(), 'downloads');
    if (!fs.existsSync(downloadPath)) {
      fs.mkdirSync(downloadPath, { recursive: true });
      console.log(`📁 Pasta downloads criada`);
    }
    
    console.log('🌐 Iniciando browser...');
    
    browser = await chromium.launch({
      headless: false,
      slowMo: 1000,
      args: ['--start-maximized', '--disable-blink-features=AutomationControlled']
    });
    
    const context = await browser.newContext({
      viewport: null,
      acceptDownloads: true,
      userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    });
    
    const page = await context.newPage();
    
    console.log('📄 Carregando página de login...');
    await page.goto('https://balcaodigital.e-redes.pt/login');
    await page.waitForLoadState('networkidle');
    
    // Aceitar cookies automaticamente
    try {
      await page.locator('text=Aceitar todos os cookies').click({ timeout: 2000 });
      console.log('🍪 Cookies aceites automaticamente');
    } catch (e) {
      console.log('🍪 Sem banner de cookies');
    }
    
    console.log('👤 Selecionando Particular...');
    await page.getByText('Particular').click();
    await page.waitForTimeout(1500);
    
    console.log('📝 Preenchendo credenciais...');
    await page.locator('input[name="username"]').fill(nif);
    await page.waitForTimeout(800);
    await page.locator('input[name="labelPassword"]').fill(password);
    await page.waitForTimeout(800);
    
    // Verificar CAPTCHA antes de fazer login
    console.log('🔍 Verificando CAPTCHA...');
    await waitForCaptcha(page);
    
    console.log('🚪 Fazendo login...');
    await page.getByRole('button', { name: 'Entrar' }).click();
    
    // Aguardar e verificar login
    console.log('⏳ Aguardando resultado do login...');
    await page.waitForTimeout(4000);
    
    const loginSuccess = await checkLoginSuccess(page);
    
    if (!loginSuccess) {
      console.log('❌ Login falhou - credenciais incorretas ou erro no site');
      throw new Error('Login falhou');
    }
    
    console.log('✅ Login bem-sucedido!');
    
    // Continuar automaticamente
    console.log('🏠 Navegando para home...');
    await page.goto('https://balcaodigital.e-redes.pt/home');
    await page.waitForLoadState('networkidle');
    
    console.log('📊 Procurando secção de consumos...');
    
    // Tentar várias formas de encontrar "Os meus locais"
    const locaisSelectors = [
      'text=Os meus locais',
      'nz-card >> text=Os meus locais',
      '[class*="card"] >> text=Os meus locais'
    ];
    
    let clicked = false;
    for (const selector of locaisSelectors) {
      try {
        await page.locator(selector).first().click({ timeout: 3000 });
        console.log('✅ "Os meus locais" encontrado');
        clicked = true;
        break;
      } catch (e) {
        continue;
      }
    }
    
    if (!clicked) {
      throw new Error('Não foi possível encontrar "Os meus locais"');
    }
    
    await page.waitForTimeout(2000);
    
    console.log('⚡ Selecionando consumos e potências...');
    await page.getByText('Produção, consumos e potências').click();
    await page.waitForTimeout(3000);
    
    console.log('📈 Abrindo consultar histórico...');
    await page.locator('text=Consultar histórico').click();
    await page.waitForTimeout(3000);
    
    console.log(`🔌 Procurando CPE ${cleanCPE}...`);
    await page.waitForTimeout(2000);
    
    // Tentar encontrar e clicar no CPE
    const cpeSelectors = [
      `text=${cleanCPE}`,
      `[class*="listitem"] >> text=${cleanCPE}`,
      `li >> text=${cleanCPE}`
    ];
    
    let cpeFound = false;
    for (const selector of cpeSelectors) {
      try {
        await page.locator(selector).first().click({ timeout: 3000 });
        console.log('✅ CPE selecionado');
        cpeFound = true;
        break;
      } catch (e) {
        continue;
      }
    }
    
    if (!cpeFound) {
      throw new Error(`CPE ${cleanCPE} não encontrado`);
    }
    
    await page.waitForTimeout(2000);
    
    console.log(`📅 Selecionando mês ${month}...`);
    await page.getByRole('textbox', { name: 'Selecionar mês' }).click();
    await page.waitForTimeout(1000);
    await page.getByText(month).first().click();
    await page.waitForTimeout(2000);
    
    console.log('📊 Iniciando exportação...');
    await page.locator('text=Exportar excel').click();
    await page.waitForTimeout(1500);
    
    // Setup e executar download
    const downloadPromise = page.waitForEvent('download');
    console.log('⬇️ Exportando...');
    await page.getByRole('button', { name: 'Exportar' }).click();
    
    console.log('⏳ Aguardando download...');
    const download = await downloadPromise;
    
    // Guardar ficheiro
    const timestamp = new Date().toISOString().split('T')[0];
    const fileName = `eredes-${nif}-${month}-${timestamp}.xlsx`;
    const filePath = path.join(downloadPath, fileName);
    
    await download.saveAs(filePath);
    
    console.log(`\n🎉 SUCESSO COMPLETO!`);
    console.log(`📁 Ficheiro guardado: ${fileName}`);
    console.log(`📍 Localização: ${downloadPath}`);
    
    if (fs.existsSync(filePath)) {
      const stats = fs.statSync(filePath);
      console.log(`📊 Tamanho: ${(stats.size / 1024).toFixed(1)} KB`);
    }
    
    console.log('\n✅ Processo automático concluído com sucesso!');
    
    // Aguardar 3 segundos antes de fechar
    console.log('⏳ Fechando browser em 3 segundos...');
    await page.waitForTimeout(3000);
    
  } catch (error) {
    console.error('\n❌ ERRO:', error.message);
    console.log('\n🔧 Possíveis soluções:');
    console.log('1. Verifique as credenciais (NIF/Password)');
    console.log('2. Verifique se o CPE está correto');
    console.log('3. Verifique se o mês existe (mai, jun, jul, etc.)');
    console.log('4. Tente novamente em alguns minutos');
    
    await page?.waitForTimeout(5000);
  } finally {
    if (browser) {
      await browser.close();
    }
    rl.close();
    console.log('\n👋 Bot terminado.');
    process.exit(0);
  }
}

console.log('🤖 E-REDES Automático v4.0');
console.log('===========================');
console.log('✨ Versão totalmente automática');
console.log('🔒 Só para para CAPTCHA se necessário\n');

extractERedesDataAuto();