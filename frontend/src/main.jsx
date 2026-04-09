import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { ClerkProvider } from '@clerk/clerk-react'
import './index.css'
import App from './App.jsx'

const clerkPublishableKey = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY

createRoot(document.getElementById('root')).render(
  <StrictMode>
    {clerkPublishableKey ? (
      <ClerkProvider publishableKey={clerkPublishableKey}>
        <App />
      </ClerkProvider>
    ) : (
      <div style={{ padding: '2rem', color: '#fff', background: '#0f172a', minHeight: '100vh' }}>
        <h2>Clerk is not configured</h2>
        <p>Set <code>VITE_CLERK_PUBLISHABLE_KEY</code> in the frontend environment and restart Vite.</p>
      </div>
    )}
  </StrictMode>,
)
