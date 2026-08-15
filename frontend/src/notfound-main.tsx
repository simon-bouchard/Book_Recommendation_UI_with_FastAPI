import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import NotFoundApp from './NotFoundApp'
import './index.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <NotFoundApp />
  </StrictMode>,
)
