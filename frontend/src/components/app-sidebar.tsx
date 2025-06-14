"use client"

import * as React from "react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { DollarSign, Menu, X } from "lucide-react"

import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarRail,
} from "@/components/ui/sidebar"

import { menuItems, tradingItems, settingsItems } from "@/config/navigation"
import { useBreakpoints } from "@/hooks/use-responsive"
import { Button } from "./ui/button"
import { cn } from "@/lib/utils"

export function AppSidebar({ className, ...props }: React.ComponentProps<typeof Sidebar>) {
  const pathname = usePathname()
  const { atLeast, isMobile, isTablet } = useBreakpoints()
  const [isOpen, setIsOpen] = React.useState(false)
  
  // Close mobile menu when navigating
  React.useEffect(() => {
    if (isMobile) {
      setIsOpen(false)
    }
  }, [pathname, isMobile])

  // Automatically open sidebar on larger screens
  React.useEffect(() => {
    if (atLeast('lg')) {
      setIsOpen(true)
    } else if (isMobile) {
      setIsOpen(false)
    }
  }, [atLeast, isMobile])

  return (
    <>
      {/* Mobile menu button - only visible on small screens */}
      <div className="fixed top-4 left-4 z-50 md:hidden">
        <Button
          variant="outline"
          size="icon"
          onClick={() => setIsOpen(!isOpen)}
          aria-label={isOpen ? "Close menu" : "Open menu"}
          className="bg-background/80 backdrop-blur-sm shadow-sm"
        >
          {isOpen ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
        </Button>
      </div>

      {/* Overlay for mobile - only visible when mobile menu is open */}
      {(isMobile || isTablet) && isOpen && (
        <div 
          className="fixed inset-0 bg-background/80 backdrop-blur-sm z-40 md:bg-transparent md:backdrop-blur-none" 
          onClick={() => setIsOpen(false)}
          aria-hidden="true"
        />
      )}

      <Sidebar 
        className={cn(
          // Base styles
          "transition-all duration-300 z-40",
          
          // Mobile styles: fixed positioning, off-canvas by default
          "fixed top-0 left-0 h-full",
          "w-[280px] md:w-auto",
          
          // Conditional positioning based on breakpoint and state
          isMobile && !isOpen && "-translate-x-full",
          isMobile && isOpen && "translate-x-0 shadow-xl",
          
          // Tablet+ styles: conditional width based on isOpen
          !isMobile && !isOpen && "w-[70px]",
          !isMobile && isOpen && "md:w-[240px] lg:w-[280px]",
          
          // Desktop styles: always visible but may be collapsed
          "md:relative md:translate-x-0",
          
          // Custom classes
          className
        )}
        collapsible={isMobile ? "offcanvas" : "icon"}
        {...props}
      >
        <SidebarHeader>
          <div className="flex items-center gap-2 px-4 py-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary text-primary-foreground shrink-0">
              <DollarSign className="h-4 w-4" />
            </div>
            <div className="grid flex-1 text-left text-sm leading-tight">
              <span className="truncate font-semibold">SmartMarketOOPS</span>
              <span className="truncate text-xs text-muted-foreground">
                AI Trading Platform
              </span>
            </div>
            
            {/* Close button for mobile - only visible on small screens when menu is open */}
            {isMobile && (
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setIsOpen(false)}
                className="md:hidden"
                aria-label="Close menu"
              >
                <X className="h-4 w-4" />
              </Button>
            )}
          </div>
        </SidebarHeader>
        
        <SidebarContent className="overflow-y-auto">
          {/* Main Navigation */}
          <SidebarGroup>
            <SidebarGroupLabel className="text-xs hidden md:block">Navigation</SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu>
                {menuItems.map((item) => (
                  <SidebarMenuItem key={item.title}>
                    <SidebarMenuButton 
                      asChild 
                      isActive={pathname === item.url}
                      tooltip={item.description}
                      className={cn(
                        "py-2",
                        isMobile && "px-4"
                      )}
                    >
                      <Link href={item.url}>
                        <item.icon className="h-4 w-4" />
                        <span className={cn(
                          "text-sm ml-3",
                          !isOpen && !isMobile && "hidden md:hidden"
                        )}>
                          {item.title}
                        </span>
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                ))}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>

          {/* Trading Section */}
          <SidebarGroup>
            <SidebarGroupLabel className="text-xs hidden md:block">Trading</SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu>
                {tradingItems.map((item) => (
                  <SidebarMenuItem key={item.title}>
                    <SidebarMenuButton 
                      asChild 
                      isActive={pathname === item.url}
                      tooltip={item.description}
                      className={cn(
                        "py-2",
                        isMobile && "px-4"
                      )}
                    >
                      <Link href={item.url}>
                        <item.icon className="h-4 w-4" />
                        <span className={cn(
                          "text-sm ml-3",
                          !isOpen && !isMobile && "hidden md:hidden"
                        )}>
                          {item.title}
                        </span>
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                ))}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>

          {/* Settings Section */}
          <SidebarGroup>
            <SidebarGroupLabel className="text-xs hidden md:block">Settings</SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu>
                {settingsItems.map((item) => (
                  <SidebarMenuItem key={item.title}>
                    <SidebarMenuButton 
                      asChild 
                      isActive={pathname === item.url}
                      tooltip={item.description}
                      className={cn(
                        "py-2",
                        isMobile && "px-4"
                      )}
                    >
                      <Link href={item.url}>
                        <item.icon className="h-4 w-4" />
                        <span className={cn(
                          "text-sm ml-3",
                          !isOpen && !isMobile && "hidden md:hidden"
                        )}>
                          {item.title}
                        </span>
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                ))}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        </SidebarContent>
        
        <SidebarFooter>
          <div className="p-4 text-xs text-muted-foreground">
            <div className="flex items-center gap-2">
              <div className="h-2 w-2 rounded-full bg-green-500 shrink-0" />
              <span className={cn(
                !isOpen && !isMobile && "hidden md:hidden"
              )}>
                All services online
              </span>
            </div>
          </div>
        </SidebarFooter>
        
        {!isMobile && <SidebarRail />}
      </Sidebar>
    </>
  )
}
