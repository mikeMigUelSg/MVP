const express = require('express');
const session = require('express-session');
const cors = require('cors');
const authRouter = require('./routes/auth');

const app = express();

const CLIENT_ORIGINS = (process.env.CLIENT_ORIGINS || process.env.CLIENT_ORIGIN || 'http://localhost:3000')
  .split(',')
  .map(origin => origin.trim())
  .filter(Boolean);

app.use(cors({
  origin: CLIENT_ORIGINS,
  credentials: true,
}));

app.use(express.json({ limit: '1mb' }));

const sessionSecret = process.env.SESSION_SECRET || 'change-me-in-production';

app.use(session({
  name: 'eredes.sid',
  secret: sessionSecret,
  resave: false,
  saveUninitialized: false,
  cookie: {
    httpOnly: true,
    sameSite: process.env.SESSION_SAMESITE || 'lax',
    secure: process.env.NODE_ENV === 'production',
    maxAge: 1000 * 60 * 60 * 4, // 4 hours
  },
}));

app.use('/api/auth', authRouter);

app.get('/api/health', (_req, res) => {
  res.json({ status: 'ok' });
});

app.use((err, _req, res, _next) => {
  console.error('[backend] unhandled error', err);
  const status = err.status || 500;
  res.status(status).json({
    message: err.message || 'Erro interno do servidor',
  });
});

const port = Number(process.env.PORT) || 3001;

if (require.main === module) {
  app.listen(port, () => {
    console.log(`E-REDES backend listening on port ${port}`);
  });
}

module.exports = app;
