const express = require('express');
const cors = require('cors');
const { createProxyMiddleware } = require('http-proxy-middleware');

const app = express();
const PORT = 3001;

// Enable CORS for all origins
app.use(cors({
  origin: '*',
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['*'],
  credentials: true
}));

// Proxy middleware for E-REDES login page only
app.use('/api/proxy/eredes-login', createProxyMiddleware({
  target: 'https://balcaodigital.e-redes.pt',
  changeOrigin: true,
  pathRewrite: {
    '^/api/proxy/eredes-login': '/login'
  },
  onProxyRes: (proxyRes, req, res) => {
    // Remove CSP headers that prevent iframe embedding
    delete proxyRes.headers['x-frame-options'];
    delete proxyRes.headers['content-security-policy'];
    delete proxyRes.headers['content-security-policy-report-only'];

    // Add headers to allow iframe embedding
    proxyRes.headers['X-Frame-Options'] = 'ALLOWALL';

    // Allow all origins for iframe
    proxyRes.headers['Access-Control-Allow-Origin'] = '*';
    proxyRes.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS';
    proxyRes.headers['Access-Control-Allow-Headers'] = '*';

    // Intercept HTML response to fix asset URLs
    if (proxyRes.headers['content-type'] && proxyRes.headers['content-type'].includes('text/html')) {
      let body = '';
      const originalWrite = res.write;
      const originalEnd = res.end;

      res.write = function(chunk) {
        body += chunk;
      };

      res.end = function(chunk) {
        if (chunk) body += chunk;

        // Replace relative asset URLs with absolute E-REDES URLs
        body = body
          .replace(/src="\/([^"]+\.js[^"]*)"/g, 'src="https://balcaodigital.e-redes.pt/$1"')
          .replace(/href="\/([^"]+\.css[^"]*)"/g, 'href="https://balcaodigital.e-redes.pt/$1"')
          .replace(/src="\/([^"]+\.(png|jpg|jpeg|gif|svg)[^"]*)"/g, 'src="https://balcaodigital.e-redes.pt/$1"');

        res.write = originalWrite;
        res.end = originalEnd;
        res.end(body);
      };
    }
  },
  onError: (err, req, res) => {
    console.error('Proxy error:', err);
    res.status(500).json({ error: 'Proxy error', details: err.message });
  }
}));

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'ok', message: 'Proxy server is running' });
});

// Start the server
app.listen(PORT, () => {
  console.log(`🔗 Proxy server running on http://localhost:${PORT}`);
  console.log(`📍 E-REDES login proxy: http://localhost:${PORT}/api/proxy/eredes-login`);
});