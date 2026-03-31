import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import ProtectedRoute from './components/ProtectedRoute';
import LandingPage from './pages/LandingPage';
import AuthPage from './pages/AuthPage';
import SheetsListPage from './pages/SheetsListPage';
import SheetDetailPage from './pages/SheetDetailPage';
import App from './App.tsx';
import './index.css';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/login" element={<AuthPage />} />
          <Route path="/dashboard" element={<ProtectedRoute><SheetsListPage /></ProtectedRoute>} />
          <Route path="/record" element={<ProtectedRoute><App /></ProtectedRoute>} />
          <Route path="/sheet/:id" element={<ProtectedRoute><SheetDetailPage /></ProtectedRoute>} />
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  </StrictMode>,
);
