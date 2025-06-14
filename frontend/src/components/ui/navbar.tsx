"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { Settings, BarChart3 } from "lucide-react";

const Navbar = () => {
  const pathname = usePathname();

  return (
    <div className="flex items-center space-x-4">
      <Link href="/analytics" className={cn(
        "flex items-center text-sm font-medium transition-colors hover:text-primary",
        pathname === "/analytics"
          ? "text-primary"
          : "text-muted-foreground"
      )}>
        <BarChart3 className="mr-2 h-4 w-4" />
        Analytics
      </Link>
      <Link href="/settings" className={cn(
        "flex items-center text-sm font-medium transition-colors hover:text-primary",
        pathname === "/settings"
          ? "text-primary"
          : "text-muted-foreground"
      )}>
        <Settings className="mr-2 h-4 w-4" />
        Settings
      </Link>
    </div>
  );
};

export default Navbar; 