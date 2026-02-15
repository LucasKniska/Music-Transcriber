import axios from 'axios';
import type { RenderedNote, SessionPayload } from '../types'; // Import new type

const API_BASE_URL = 'http://3.94.194.252:5000/api';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// --- FETCHING (LOAD ON START) ---
export const fetchNotes = async (): Promise<RenderedNote[]> => {
  try {
    // Assuming backend returns the last session's notes array
    const response = await apiClient.get<RenderedNote[]>('/notes'); 
    return response.data;
  } catch (error) {
    console.error('API Error fetching notes:', error);
    return []; 
  }
};

// --- BATCH SAVING (NEW) ---
export const saveSession = async (sessionData: SessionPayload) => {
  // Post the entire object to your backend
  const response = await apiClient.post('/sessions', sessionData);
  console.log(response.data);
  return response.data;
};

export const fetchPDFExport = async (): Promise<Blob> => {
  const response = await apiClient.get('/export', { 
    responseType: 'blob', 
  });
  return response.data;
};

export const clearAllNotes = async () => {
  const response = await apiClient.delete('/notes');
  return response.data;
};