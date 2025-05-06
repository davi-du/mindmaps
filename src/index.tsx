import React from 'react'
import ReactDOM from 'react-dom/client'
import { createBrowserRouter, RouterProvider } from 'react-router-dom'

import { ChatApp } from './App'

// ! the only css imports in ts/x files
import './css/index.scss'
import 'reactflow/dist/style.css'

if (process.env.NODE_ENV === 'production') {
  console.log = () => {}
  console.error = () => {}
  console.debug = () => {}
}

if (typeof window !== 'undefined') {
  const observerError = console.error;
  console.error = (...args) => {
    if (
      typeof args[0] === 'string' &&
      args[0].includes('ResizeObserver loop completed with undelivered notifications')
    ) {
      return;
    }
    observerError(...args);
  };
}

/* -------------------------------------------------------------------------- */

// ! set up react router
const router = createBrowserRouter([
  {
    path: '/*',
    element: <ChatApp />,
  },
])

// ! render app
const root = ReactDOM.createRoot(document.getElementById('root')!)
root.render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>,
)
