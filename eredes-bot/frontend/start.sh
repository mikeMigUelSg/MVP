#!/bin/bash

# E-REDES Bot Frontend - Script de Inicialização
echo "🚀 E-REDES Bot Frontend - Inicializando..."
echo "=================================================="

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Verificar se Node.js está instalado
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js não encontrado!${NC}"
    echo "Por favor, instale Node.js (versão 16 ou superior) em:"
    echo "https://nodejs.org/"
    exit 1
fi

# Verificar versão do Node.js
NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 16 ]; then
    echo -e "${RED}❌ Node.js versão muito antiga!${NC}"
    echo "Versão atual: $(node -v)"
    echo "Versão mínima: v16.0.0"
    echo "Por favor, atualize o Node.js em: https://nodejs.org/"
    exit 1
fi

echo -e "${GREEN}✅ Node.js versão:${NC} $(node -v)"

# Verificar se npm está instalado
if ! command -v npm &> /dev/null; then
    echo -e "${RED}❌ npm não encontrado!${NC}"
    exit 1
fi

echo -e "${GREEN}✅ npm versão:${NC} $(npm -v)"

# Verificar se estamos no diretório correto
if [ ! -f "package.json" ]; then
    echo -e "${RED}❌ package.json não encontrado!${NC}"
    echo "Por favor, execute este script no diretório frontend/"
    exit 1
fi

# Verificar se node_modules existe
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}📦 Instalando dependências...${NC}"
    npm install

    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Falha na instalação das dependências!${NC}"
        exit 1
    fi

    echo -e "${GREEN}✅ Dependências instaladas com sucesso!${NC}"
else
    echo -e "${GREEN}✅ Dependências já instaladas${NC}"
fi

# Verificar se existe .env
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠️  Arquivo .env não encontrado, usando configurações padrão${NC}"
fi

# Mostrar informações do projeto
echo ""
echo -e "${BLUE}📋 Informações do Projeto:${NC}"
echo "- Nome: E-REDES Bot Frontend"
echo "- Versão: 1.0.0"
echo "- Porta padrão: 3000"
echo "- API Backend: http://localhost:3001/api"
echo ""

# Verificar se a porta 3000 está disponível
if lsof -Pi :3000 -sTCP:LISTEN -t >/dev/null ; then
    echo -e "${YELLOW}⚠️  Porta 3000 já está em uso!${NC}"
    echo "A aplicação tentará usar uma porta alternativa."
    echo ""
fi

# Opções de execução
echo -e "${BLUE}🔧 Opções disponíveis:${NC}"
echo "1. Iniciar em modo desenvolvimento (recomendado)"
echo "2. Criar build de produção"
echo "3. Executar testes"
echo "4. Analisar bundle size"
echo ""

read -p "Escolha uma opção (1-4) [1]: " choice
choice=${choice:-1}

case $choice in
    1)
        echo -e "${GREEN}🚀 Iniciando em modo desenvolvimento...${NC}"
        echo ""
        echo "A aplicação será aberta automaticamente no navegador."
        echo "URL: http://localhost:3000"
        echo ""
        echo "Para parar o servidor, pressione Ctrl+C"
        echo ""
        echo "=================================================="
        npm start
        ;;
    2)
        echo -e "${GREEN}🏗️  Criando build de produção...${NC}"
        npm run build

        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ Build criado com sucesso em ./build/${NC}"
            echo ""
            echo "Para servir o build localmente:"
            echo "npx serve -s build -l 3000"
        else
            echo -e "${RED}❌ Falha na criação do build!${NC}"
        fi
        ;;
    3)
        echo -e "${GREEN}🧪 Executando testes...${NC}"
        npm test
        ;;
    4)
        echo -e "${GREEN}📊 Analisando tamanho do bundle...${NC}"
        npm run analyze
        ;;
    *)
        echo -e "${RED}❌ Opção inválida!${NC}"
        exit 1
        ;;
esac