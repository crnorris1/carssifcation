import React from 'react';
import { createRoot } from 'react-dom/client';
//import './index.css'; // Optional: if you had a CSS file
import App from './App'; // Assumes your main component is in src/App.js

const container = document.getElementById('root');
const root = createRoot(container); // Create a root
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);