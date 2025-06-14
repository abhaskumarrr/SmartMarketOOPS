"use client"

import * as React from "react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { Menu } from "lucide-react"

import { Button } from "@/components/ui/button"
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet"
import { menuItems, tradingItems, settingsItems } from "@/config/navigation"

export function MobileNav() {
  const [open, setOpen] = React.useState(false)
  const pathname = usePathname()

  // Close the mobile nav when changing routes
  React.useEffect(() => {
    setOpen(false)
  }, [pathname])

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          className="md:hidden"
          aria-label="Open mobile navigation"
        >
          <Menu className="h-5 w-5" />
        </Button>
      </SheetTrigger>
      <SheetContent side="left" className="w-72 px-0">
        <SheetHeader className="px-6 pt-6 pb-4 border-b">
          <SheetTitle className="text-xl font-bold">SmartMarketOOPS</SheetTitle>
        </SheetHeader>
        <div className="py-4">
          <div className="space-y-4">
            <div className="px-6">
              <h3 className="text-xs font-semibold text-muted-foreground">
                Navigation
              </h3>
              <nav className="mt-2 space-y-1">
                {menuItems.map((item) => (
                  <Link
                    key={item.title}
                    href={item.url}
                    className={`flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium ${
                      pathname === item.url
                        ? "bg-muted text-foreground"
                        : "text-muted-foreground hover:bg-muted hover:text-foreground"
                    }`}
                  >
                    <item.icon className="h-4 w-4" />
                    {item.title}
                  </Link>
                ))}
              </nav>
            </div>

            <div className="px-6">
              <h3 className="text-xs font-semibold text-muted-foreground">
                Trading
              </h3>
              <nav className="mt-2 space-y-1">
                {tradingItems.map((item) => (
                  <Link
                    key={item.title}
                    href={item.url}
                    className={`flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium ${
                      pathname === item.url
                        ? "bg-muted text-foreground"
                        : "text-muted-foreground hover:bg-muted hover:text-foreground"
                    }`}
                  >
                    <item.icon className="h-4 w-4" />
                    {item.title}
                  </Link>
                ))}
              </nav>
            </div>

            <div className="px-6">
              <nav className="space-y-1">
                {settingsItems.map((item) => (
                  <Link
                    key={item.title}
                    href={item.url}
                    className={`flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium ${
                      pathname === item.url
                        ? "bg-muted text-foreground"
                        : "text-muted-foreground hover:bg-muted hover:text-foreground"
                    }`}
                  >
                    <item.icon className="h-4 w-4" />
                    {item.title}
                  </Link>
                ))}
              </nav>
            </div>
          </div>
        </div>
      </SheetContent>
    </Sheet>
  )
} 