# Bot E-REDES Balcão Digital 🤖⚡

Bot automatizado para extrair dados de consumo energético do **Balcão Digital da E-REDES** usando Playwright com intervenção manual quando necessário.

## 🎯 Funcionalidades

- **Login automático** com NIF e password
- **Resolução manual de CAPTCHA** (human-in-the-loop)
- **Navegação automática** até aos dados de consumo
- **Seleção de CPE e mês** personalizáveis
- **Exportação automática** para Excel
- **Downloads organizados** com nomes informativos

## 🚀 Instalação Rápida

### 1. Pré-requisitos
- **Node.js** (versão 16 ou superior)
- **npm** ou **yarn**

### 2. Instalação
```bash
# Clonar ou criar pasta do projeto
mkdir eredes-bot && cd eredes-bot

# Copiar os ficheiros (eredes-bot.js e package.json)

# Instalar dependências
npm install

# Instalar browser do Playwright
npm run setup
```

### 3. Executar
```bash
npm start
```

## 📋 Como Usar

1. **Execute o comando** `npm start`
2. **Forneça os dados solicitados:**
   - NIF
   - Password
   - Código CPE (ex: `PT0002000071954119ZW`)
   - Mês (ex: `mai`, `jun`, `jul`)

3. **Intervenha quando necessário:**
   - Se aparecer CAPTCHA, resolva-o manualmente
   - Se houver problemas de login, corrija-os na janela do browser
   - Aguarde a mensagem para continuar

4. **O ficheiro será guardado** na pasta `downloads/` com o nome:
   ```
   eredes-consumo-[NIF]-[MES]-[DATA].xlsx
   ```

## 🔧 Como Funciona

### Fluxo do Bot:
1. **Abre browser** em modo visual (não headless)
2. **Navega** para o Balcão Digital
3. **Preenche credenciais** automaticamente
4. **Pausa** para resolução manual de CAPTCHA
5. **Navega** até aos dados de consumo
6. **Seleciona** CPE e mês especificados
7. **Exporta** dados para Excel
8. **Guarda ficheiro** localmente

### Pauses Estratégicas:
- ⏸️ **Após credenciais:** Para resolver CAPTCHA
- ⏸️ **Após login:** Para verificar se foi bem-sucedido  
- ⏸️ **Após download:** Para verificar resultado
- ⏸️ **Se erro:** Para resolução manual

## 🛡️ Segurança

- ✅ **Sem armazenamento:** Credenciais não são guardadas
- ✅ **Browser local:** Tudo executado na sua máquina
- ✅ **Intervenção manual:** Controlo total quando necessário
- ✅ **Código aberto:** Pode auditar todo o processo

## 📁 Estrutura de Ficheiros

```
eredes-bot/
├── eredes-bot.js      # Script principal
├── package.json       # Configuração do projeto
├── README.md         # Este ficheiro
└── downloads/        # Pasta dos ficheiros exportados (criada automaticamente)
```

## ⚠️ Notas Importantes

- **Browser visível:** O Chromium abre em modo visível para permitir intervenção
- **Velocidade controlada:** Delays entre ações para estabilidade
- **Robustez:** Trata erros e permite correção manual
- **Downloads:** Ficheiros guardados com nomes informativos

## 🐛 Resolução de Problemas

### Problema: Browser não abre
**Solução:** Execute `npm run setup` para instalar o Chromium

### Problema: Erro de login
**Solução:** Verifique credenciais ou resolva manualmente na janela do browser

### Problema: CPE não encontrado  
**Solução:** Verifique se o código CPE está correto e completo

### Problema: Mês não encontrado
**Solução:** Use abreviações portuguesas (jan, fev, mar, abr, mai, jun, jul, ago, set, out, nov, dez)

## 🤝 Contribuições

Contribuições são bem-vindas! Abra issues para bugs ou sugestões de melhorias.

## 📄 Licença

MIT License - Use livremente!