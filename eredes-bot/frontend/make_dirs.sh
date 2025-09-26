#!/usr/bin/env bash
# cria_estrutura.sh
# Uso: ./cria_estrutura.sh [DIRETORIO_BASE]
# Ex.: ./cria_estrutura.sh meu_projeto

set -euo pipefail

BASE_DIR="${1:-.}"

# --- Helpers ---
ensure_dir() {
  local d="$1"
  if [ ! -d "$d" ]; then
    mkdir -p "$d"
    echo "📁 Criado diretório: $d"
  else
    echo "✔️  Já existe: $d"
  fi
}

ensure_file() {
  local f="$1"
  local dir
  dir="$(dirname "$f")"
  ensure_dir "$dir"
  if [ ! -f "$f" ]; then
    : > "$f"   # cria ficheiro vazio
    echo "📄 Criado ficheiro: $f"
  else
    echo "✔️  Já existe: $f"
  fi
}

# --- Diretórios (alguns também serão garantidos via ensure_file) ---
DIRS=(
  "$BASE_DIR/public"
  "$BASE_DIR/src"
  "$BASE_DIR/src/components"
  "$BASE_DIR/src/components/Button"
  "$BASE_DIR/src/components/Card"
  "$BASE_DIR/src/components/Chart"
  "$BASE_DIR/src/components/Header"
  "$BASE_DIR/src/components/Input"
  "$BASE_DIR/src/components/LoadingSpinner"
  "$BASE_DIR/src/components/MetricCard"
  "$BASE_DIR/src/components/Modal"
  "$BASE_DIR/src/components/Sidebar"
  "$BASE_DIR/src/components/StatusIndicator"
  "$BASE_DIR/src/components/Tooltip"
  "$BASE_DIR/src/pages"
  "$BASE_DIR/src/pages/Dashboard"
  "$BASE_DIR/src/pages/Results"
  "$BASE_DIR/src/pages/Settings"
  "$BASE_DIR/src/pages/Simulation"
  "$BASE_DIR/src/pages/Simulation/components"
  "$BASE_DIR/src/contexts"
  "$BASE_DIR/src/hooks"
  "$BASE_DIR/src/services"
  "$BASE_DIR/src/styles"
  "$BASE_DIR/src/utils"
)

# --- Ficheiros ---
FILES=(
  "$BASE_DIR/public/index.html"
  "$BASE_DIR/public/favicon.ico"
  "$BASE_DIR/public/manifest.json"

  "$BASE_DIR/src/components/index.js"

  "$BASE_DIR/src/components/Button/Button.jsx"
  "$BASE_DIR/src/components/Button/Button.styles.css"
  "$BASE_DIR/src/components/Button/index.js"

  "$BASE_DIR/src/components/Card/Card.jsx"
  "$BASE_DIR/src/components/Card/Card.styles.css"
  "$BASE_DIR/src/components/Card/index.js"

  "$BASE_DIR/src/components/Chart/Chart.jsx"
  "$BASE_DIR/src/components/Chart/Chart.styles.css"
  "$BASE_DIR/src/components/Chart/index.js"

  "$BASE_DIR/src/components/Header/Header.jsx"
  "$BASE_DIR/src/components/Header/Header.styles.css"
  "$BASE_DIR/src/components/Header/index.js"

  "$BASE_DIR/src/components/Input/Input.jsx"
  "$BASE_DIR/src/components/Input/Input.styles.css"
  "$BASE_DIR/src/components/Input/index.js"

  "$BASE_DIR/src/components/LoadingSpinner/LoadingSpinner.jsx"
  "$BASE_DIR/src/components/LoadingSpinner/LoadingSpinner.styles.css"
  "$BASE_DIR/src/components/LoadingSpinner/index.js"

  "$BASE_DIR/src/components/MetricCard/MetricCard.jsx"
  "$BASE_DIR/src/components/MetricCard/MetricCard.styles.css"
  "$BASE_DIR/src/components/MetricCard/index.js"

  "$BASE_DIR/src/components/Modal/Modal.jsx"
  "$BASE_DIR/src/components/Modal/Modal.styles.css"
  "$BASE_DIR/src/components/Modal/index.js"

  "$BASE_DIR/src/components/Sidebar/Sidebar.jsx"
  "$BASE_DIR/src/components/Sidebar/Sidebar.styles.css"
  "$BASE_DIR/src/components/Sidebar/index.js"

  "$BASE_DIR/src/components/StatusIndicator/StatusIndicator.jsx"
  "$BASE_DIR/src/components/StatusIndicator/StatusIndicator.styles.css"
  "$BASE_DIR/src/components/StatusIndicator/index.js"

  "$BASE_DIR/src/components/Tooltip/Tooltip.jsx"
  "$BASE_DIR/src/components/Tooltip/Tooltip.styles.css"
  "$BASE_DIR/src/components/Tooltip/index.js"

  "$BASE_DIR/src/pages/Dashboard/Dashboard.jsx"
  "$BASE_DIR/src/pages/Dashboard/Dashboard.styles.css"
  "$BASE_DIR/src/pages/Dashboard/index.js"

  "$BASE_DIR/src/pages/Results/Results.jsx"
  "$BASE_DIR/src/pages/Results/Results.styles.css"
  "$BASE_DIR/src/pages/Results/index.js"

  "$BASE_DIR/src/pages/Settings/Settings.jsx"
  "$BASE_DIR/src/pages/Settings/Settings.styles.css"
  "$BASE_DIR/src/pages/Settings/index.js"

  "$BASE_DIR/src/pages/Simulation/Simulation.jsx"
  "$BASE_DIR/src/pages/Simulation/Simulation.styles.css"
  "$BASE_DIR/src/pages/Simulation/components/ConfigurationPanel.jsx"
  "$BASE_DIR/src/pages/Simulation/components/LoginForm.jsx"
  "$BASE_DIR/src/pages/Simulation/index.js"

  "$BASE_DIR/src/contexts/AuthContext.jsx"
  "$BASE_DIR/src/contexts/ThemeContext.jsx"
  "$BASE_DIR/src/contexts/SimulationContext.jsx"

  "$BASE_DIR/src/hooks/useApi.js"
  "$BASE_DIR/src/hooks/useLocalStorage.js"
  "$BASE_DIR/src/hooks/useDebounce.js"

  "$BASE_DIR/src/services/api.js"
  "$BASE_DIR/src/services/eredesService.js"
  "$BASE_DIR/src/services/authService.js"

  "$BASE_DIR/src/styles/GlobalStyles.js"
  "$BASE_DIR/src/styles/variables.css"
  "$BASE_DIR/src/styles/animations.css"

  "$BASE_DIR/src/utils/dateHelpers.js"
  "$BASE_DIR/src/utils/formatters.js"
  "$BASE_DIR/src/utils/constants.js"

  "$BASE_DIR/src/App.jsx"
  "$BASE_DIR/src/index.js"
  "$BASE_DIR/src/index.css"

  "$BASE_DIR/package.json"
  "$BASE_DIR/README.md"
  "$BASE_DIR/.gitignore"
)

echo "==> A criar diretórios (se necessário)…"
for d in "${DIRS[@]}"; do
  ensure_dir "$d"
done

echo "==> A criar ficheiros (se necessário)…"
for f in "${FILES[@]}"; do
  ensure_file "$f"
done

echo "✅ Estrutura garantida em: $BASE_DIR"