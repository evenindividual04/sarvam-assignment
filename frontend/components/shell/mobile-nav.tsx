"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { MenuIcon } from "lucide-react";
import {
  Sheet,
  SheetContent,
  SheetTrigger,
  SheetTitle,
} from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/", label: "Chat" },
  { href: "/sessions", label: "Sessions" },
  { href: "/eval", label: "Eval" },
  { href: "/settings", label: "Settings" },
];

export function MobileNav() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  return (
    <div className="md:hidden flex items-center gap-3 px-5 h-14 border-b border-border sticky top-0 bg-background/80 backdrop-blur-sm z-20">
      <Sheet open={open} onOpenChange={setOpen}>
        <SheetTrigger
          render={
            <Button variant="ghost" size="icon">
              <MenuIcon className="size-5" />
            </Button>
          }
        />

        <SheetContent
          side="left"
          className="w-[260px] p-0 bg-background border-r border-border"
        >
          <SheetTitle className="sr-only">Navigation</SheetTitle>
          <div className="px-6 pt-6 pb-5 border-b border-border">
            <div className="text-[15px] font-medium tracking-tight">
              Deep Research
            </div>
          </div>
          <nav className="px-6 py-5 flex flex-col gap-0.5">
            {NAV.map((item) => {
              const active =
                item.href === "/"
                  ? pathname === "/"
                  : pathname?.startsWith(item.href);
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  onClick={() => setOpen(false)}
                  className={cn(
                    "py-2 font-sans text-[13px] uppercase tracking-[0.12em] transition-colors",
                    active
                      ? "text-foreground"
                      : "text-muted-foreground hover:text-foreground",
                  )}
                >
                  {item.label}
                </Link>
              );
            })}
          </nav>
        </SheetContent>
      </Sheet>
      <div className="flex flex-col">
        <span className="text-sm font-medium tracking-tight leading-none">
          Deep Research
        </span>
      </div>
    </div>
  );
}
