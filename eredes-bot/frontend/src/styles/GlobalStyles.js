export const GlobalStyles = () => (
    <style jsx global>{`
      :root {
        /* Advanced Color System */
        --primary-50: #e3f2fd;
        --primary-100: #bbdefb;
        --primary-200: #90caf9;
        --primary-300: #64b5f6;
        --primary-400: #42a5f5;
        --primary-500: #2196f3;
        --primary-600: #1e88e5;
        --primary-700: #1976d2;
        --primary-800: #1565c0;
        --primary-900: #0d47a1;
        
        --success-50: #e8f5e8;
        --success-500: #4caf50;
        --success-600: #43a047;
        
        --warning-50: #fff8e1;
        --warning-500: #ff9800;
        --warning-600: #f57c00;
        
        --error-50: #ffebee;
        --error-500: #f44336;
        --error-600: #e53935;
        
        --neutral-0: #ffffff;
        --neutral-50: #fafafa;
        --neutral-100: #f5f5f5;
        --neutral-200: #eeeeee;
        --neutral-300: #e0e0e0;
        --neutral-400: #bdbdbd;
        --neutral-500: #9e9e9e;
        --neutral-600: #757575;
        --neutral-700: #616161;
        --neutral-800: #424242;
        --neutral-900: #212121;
        
        /* Glass Morphism */
        --glass-bg: rgba(255, 255, 255, 0.25);
        --glass-border: rgba(255, 255, 255, 0.18);
        --glass-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        --glass-backdrop: blur(20px);
        
        /* Advanced Spacing System */
        --space-0: 0;
        --space-1: 0.25rem;
        --space-2: 0.5rem;
        --space-3: 0.75rem;
        --space-4: 1rem;
        --space-5: 1.25rem;
        --space-6: 1.5rem;
        --space-8: 2rem;
        --space-10: 2.5rem;
        --space-12: 3rem;
        --space-16: 4rem;
        --space-20: 5rem;
        --space-24: 6rem;
        
        /* Typography Scale */
        --text-xs: 0.75rem;
        --text-sm: 0.875rem;
        --text-base: 1rem;
        --text-lg: 1.125rem;
        --text-xl: 1.25rem;
        --text-2xl: 1.5rem;
        --text-3xl: 1.875rem;
        --text-4xl: 2.25rem;
        --text-5xl: 3rem;
        
        /* Border Radius */
        --radius-none: 0;
        --radius-sm: 0.125rem;
        --radius-base: 0.25rem;
        --radius-md: 0.375rem;
        --radius-lg: 0.5rem;
        --radius-xl: 0.75rem;
        --radius-2xl: 1rem;
        --radius-3xl: 1.5rem;
        --radius-full: 9999px;
        
        /* Shadows */
        --shadow-xs: 0 1px 2px 0 rgb(0 0 0 / 0.05);
        --shadow-sm: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1);
        --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
        --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
        --shadow-2xl: 0 25px 50px -12px rgb(0 0 0 / 0.25);
        
        /* Gradients */
        --gradient-primary: linear-gradient(135deg, var(--primary-500), var(--primary-700));
        --gradient-success: linear-gradient(135deg, var(--success-500), var(--success-600));
        --gradient-glass: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
        
        /* Animations */
        --duration-75: 75ms;
        --duration-100: 100ms;
        --duration-150: 150ms;
        --duration-200: 200ms;
        --duration-300: 300ms;
        --duration-500: 500ms;
        --duration-700: 700ms;
        --duration-1000: 1000ms;
        
        --ease-linear: linear;
        --ease-in: cubic-bezier(0.4, 0, 1, 1);
        --ease-out: cubic-bezier(0, 0, 0.2, 1);
        --ease-in-out: cubic-bezier(0.4, 0, 0.2, 1);
      }
  
      * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
      }
  
      html {
        font-size: 16px;
        scroll-behavior: smooth;
      }
  
      body {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
        color: var(--neutral-800);
        line-height: 1.6;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
      }
  
      .app {
        min-height: 100vh;
        display: flex;
        flex-direction: column;
      }
  
      .app-body {
        display: flex;
        flex: 1;
        overflow: hidden;
      }
  
      .main-content {
        flex: 1;
        padding: var(--space-6);
        overflow-y: auto;
        background: rgba(255, 255, 255, 0.05);
      }
  
      /* Scrollbar Styling */
      ::-webkit-scrollbar {
        width: 8px;
      }
  
      ::-webkit-scrollbar-track {
        background: var(--neutral-100);
      }
  
      ::-webkit-scrollbar-thumb {
        background: var(--neutral-400);
        border-radius: var(--radius-full);
      }
  
      ::-webkit-scrollbar-thumb:hover {
        background: var(--neutral-500);
      }
  
      /* Focus Styles */
      *:focus {
        outline: none;
        box-shadow: 0 0 0 3px rgba(33, 150, 243, 0.3);
      }
  
      /* Utility Classes */
      .glass {
        background: var(--glass-bg);
        backdrop-filter: var(--glass-backdrop);
        -webkit-backdrop-filter: var(--glass-backdrop);
        border: 1px solid var(--glass-border);
        box-shadow: var(--glass-shadow);
      }
  
      .gradient-text {
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
      }
  
      .animate-pulse {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
      }
  
      @keyframes pulse {
        0%, 100% {
          opacity: 1;
        }
        50% {
          opacity: .5;
        }
      }
  
      .animate-bounce {
        animation: bounce 1s infinite;
      }
  
      @keyframes bounce {
        0%, 100% {
          transform: translateY(-25%);
          animation-timing-function: cubic-bezier(0.8, 0, 1, 1);
        }
        50% {
          transform: none;
          animation-timing-function: cubic-bezier(0, 0, 0.2, 1);
        }
      }
  
      .animate-fade-in {
        animation: fadeIn 0.5s ease-out;
      }
  
      @keyframes fadeIn {
        from {
          opacity: 0;
          transform: translateY(10px);
        }
        to {
          opacity: 1;
          transform: translateY(0);
        }
      }
  
      .animate-slide-up {
        animation: slideUp 0.3s ease-out;
      }
  
      @keyframes slideUp {
        from {
          transform: translateY(100%);
        }
        to {
          transform: translateY(0);
        }
      }
    `}</style>
  );
  
  // src/contexts/ThemeContext.jsx
  import React, { createContext, useContext, useState, useEffect } from 'react';
  
  const ThemeContext = createContext();
  
  export const useTheme = () => {
    const context = useContext(ThemeContext);
    if (!context) {
      throw new Error('useTheme must be used within a ThemeProvider');
    }
    return context;
  };
  
  export const ThemeProvider = ({ children }) => {
    const [isDark, setIsDark] = useState(false);
  
    useEffect(() => {
      const saved = localStorage.getItem('theme');
      if (saved) {
        setIsDark(saved === 'dark');
      }
    }, []);
  
    const toggleTheme = () => {
      const newTheme = !isDark;
      setIsDark(newTheme);
      localStorage.setItem('theme', newTheme ? 'dark' : 'light');
    };
  
    const value = {
      isDark,
      toggleTheme,
    };
  
    return (
      <ThemeContext.Provider value={value}>
        {children}
      </ThemeContext.Provider>
    );
  };