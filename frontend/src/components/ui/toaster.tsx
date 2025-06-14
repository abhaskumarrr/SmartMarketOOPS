"use client"

import { useState, useEffect } from "react"

interface ToasterProps {
  className?: string
}

interface Toast {
  id: string
  title?: string
  description: string
  variant?: "default" | "destructive" | "success" | "info"
  duration?: number
}

export const toasts: Toast[] = []

export const Toaster = ({ className }: ToasterProps) => {
  const [visibleToasts, setVisibleToasts] = useState<Toast[]>([])

  useEffect(() => {
    // Simple toast system until we can add @radix-ui/react-toast
    const handleNewToast = () => {
      setVisibleToasts([...toasts])
      
      // Clear toasts after their duration
      toasts.forEach((toast) => {
        const duration = toast.duration || 3000
        setTimeout(() => {
          setVisibleToasts((current) => 
            current.filter((t) => t.id !== toast.id)
          )
          // Remove from global toasts array
          const index = toasts.findIndex((t) => t.id === toast.id)
          if (index > -1) {
            toasts.splice(index, 1)
          }
        }, duration)
      })
    }

    // Check for new toasts every 300ms
    const interval = setInterval(() => {
      if (toasts.length > 0 && toasts.some(t => !visibleToasts.includes(t))) {
        handleNewToast()
      }
    }, 300)

    return () => clearInterval(interval)
  }, [visibleToasts])

  if (visibleToasts.length === 0) return null

  return (
    <div className="fixed bottom-0 right-0 z-50 flex flex-col items-end p-4 space-y-2">
      {visibleToasts.map((toast) => (
        <div
          key={toast.id}
          className={`p-4 rounded-md shadow-lg border animate-in slide-in-from-right max-w-md ${
            toast.variant === "destructive"
              ? "bg-red-50 dark:bg-red-900/50 border-red-400 text-red-800 dark:text-red-100"
              : toast.variant === "success"
              ? "bg-green-50 dark:bg-green-900/50 border-green-400 text-green-800 dark:text-green-100"
              : toast.variant === "info"
              ? "bg-blue-50 dark:bg-blue-900/50 border-blue-400 text-blue-800 dark:text-blue-100"
              : "bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700"
          }`}
        >
          {toast.title && (
            <div className="font-medium mb-1">{toast.title}</div>
          )}
          <div className="text-sm">{toast.description}</div>
        </div>
      ))}
    </div>
  )
}

// Simple toast function until we can fully implement the shadcn/ui version
export const toast = {
  id: 0,
  
  _createToast(options: Omit<Toast, "id">) {
    const id = String(this.id++)
    const toast = { ...options, id }
    toasts.push(toast)
    return toast
  },
  
  default(options: { title?: string; description: string; duration?: number }) {
    return this._createToast({ ...options, variant: "default" })
  },
  
  destructive(options: { title?: string; description: string; duration?: number }) {
    return this._createToast({ ...options, variant: "destructive" })
  },
  
  success(options: { title?: string; description: string; duration?: number }) {
    return this._createToast({ ...options, variant: "success" })
  },
  
  info(options: { title?: string; description: string; duration?: number }) {
    return this._createToast({ ...options, variant: "info" })
  },
} 