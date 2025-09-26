const express = require('express');
const { z } = require('zod');
const { loginToERedes } = require('../services/eredesAuth');

const router = express.Router();

const credentialsSchema = z.object({
  nif: z
    .string({ required_error: 'NIF é obrigatório' })
    .trim()
    .min(1, 'NIF é obrigatório')
    .regex(/^\d{9}$/u, 'NIF deve conter 9 dígitos'),
  password: z
    .string({ required_error: 'Password é obrigatória' })
    .min(1, 'Password é obrigatória'),
});

function sanitizeUser({ nif }) {
  return {
    nif,
    maskedNif: `${nif.slice(0, 3)}******${nif.slice(-2)}`,
  };
}

router.post('/login', async (req, res, next) => {
  try {
    const parsed = credentialsSchema.parse({
      nif: req.body?.nif ?? req.body?.username ?? '',
      password: req.body?.password ?? '',
    });

    const authData = await loginToERedes(parsed);

    req.session.auth = {
      token: authData.authToken,
      cookies: authData.cookieHeader,
      fetchedAt: new Date().toISOString(),
      user: sanitizeUser(parsed),
    };

    res.json({
      success: true,
      authToken: authData.authToken,
      cookies: authData.cookieHeader,
      user: req.session.auth.user,
    });
  } catch (error) {
    if (error instanceof z.ZodError) {
      return res.status(400).json({
        message: error.errors[0]?.message || 'Dados inválidos',
      });
    }

    if (error.name === 'InvalidCredentialsError') {
      return res.status(401).json({ message: error.message });
    }

    if (error.name === 'CaptchaError') {
      return res.status(503).json({ message: error.message });
    }

    return next(error);
  }
});

router.post('/logout', (req, res) => {
  req.session.destroy(() => {
    res.json({ success: true });
  });
});

router.get('/session', (req, res) => {
  if (req.session?.auth?.token) {
    return res.json({
      isAuthenticated: true,
      user: req.session.auth.user,
      authToken: req.session.auth.token,
      fetchedAt: req.session.auth.fetchedAt,
    });
  }

  return res.json({ isAuthenticated: false });
});

router.get('/token', (req, res) => {
  if (req.session?.auth?.token) {
    return res.json({
      authToken: req.session.auth.token,
      cookies: req.session.auth.cookies,
      fetchedAt: req.session.auth.fetchedAt,
    });
  }
  return res.status(401).json({ message: 'Sessão expirada ou não autenticada' });
});

module.exports = router;
