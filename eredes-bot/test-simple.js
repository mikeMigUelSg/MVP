// Versão simplificada para testes rápidos
const { chromium } = require('playwright');

async function quickTest() {
  console.log('🧪 Teste rápido - E-REDES Bot');
  
  // Credenciais hardcoded para teste (ALTERE AQUI)
  const CREDENTIALS = {
    nif: '193173611',        // ⚠️ ALTERE para o seu NIF
    password: 'SuaPassword', // ⚠️ ALTERE para a sua password
    cpe: 'PT0002000071954119ZW', // ⚠️ ALTERE para o seu CPE
    month: 'mai'             // ⚠️ ALTERE para o mês desejado
  };
  
  const browser = await chromium.launch({ 
    headless: false, 
    slowMo: 2000 
  });
  
  const page = await browser.newPage();
  
  try {
    // Login
    await page.goto('https://balcaodigital.e-redes.pt/login');
    await page.getByText('Particular').click();
    await page.getByRole('textbox', { name: 'NIF' }).fill(CREDENTIALS.nif);
    await page.getByText('Password', { exact: true }).click();
    await page.getByRole('textbox', { name: 'Password' }).fill(CREDENTIALS.password);
    
    console.log('⏸️ Resolva o CAPTCHA se aparecer...');
    await page.waitForTimeout(5000); // 5 segundos para CAPTCHA
    
    await page.getByRole('button', { name: 'Entrar' }).click();
    await page.waitForTimeout(5000);
    
    // Navegação
    await page.goto('https://balcaodigital.e-redes.pt/home');
    await page.locator('nz-card').filter({ hasText: '> Os meus locais Leituras,' }).click();
    await page.getByText('Produção, consumos e potências').click();
    await page.locator('div').filter({ hasText: 'Consultar histórico' }).nth(3).click();
    
    // Seleção e exportação
    await page.getByRole('listitem').filter({ hasText: `CPE Consumo${CREDENTIALS.cpe}` }).click();
    await page.getByRole('textbox', { name: 'Selecionar mês' }).click();
    await page.getByText(CREDENTIALS.month).click();
    await page.locator('a').filter({ hasText: 'Exportar excel' }).click();
    
    const downloadPromise = page.waitForEvent('download');
    await page.getByRole('button', { name: 'Exportar' }).click();
    const download = await downloadPromise;
    
    console.log('✅ Download iniciado!');
    console.log('Download path:', await download.path());
    
  } catch (error) {
    console.error('❌ Erro:', error.message);
  }
  
  await page.waitForTimeout(10000); // 10 segundos para verificar resultado
  await browser.close();
}

quickTest();