import { useCallback, useEffect, useState } from "react";

export type Theme = "light" | "dark";

function systemTheme(): Theme {
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function savedTheme(): Theme | null {
  try {
    const t = localStorage.getItem("theme");
    return t === "light" || t === "dark" ? t : null;
  } catch {
    return null;
  }
}

/** The active theme: the user's saved choice, else the OS setting. Toggling stamps data-theme on <html>. */
export function useTheme() {
  const [theme, setTheme] = useState<Theme>(() => savedTheme() ?? systemTheme());

  useEffect(() => {
    if (savedTheme()) return;
    const mq = window.matchMedia?.("(prefers-color-scheme: dark)");
    const onChange = () => setTheme(systemTheme());
    mq?.addEventListener?.("change", onChange);
    return () => mq?.removeEventListener?.("change", onChange);
  }, []);

  const toggle = useCallback(() => {
    setTheme((current) => {
      const next: Theme = current === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      try {
        localStorage.setItem("theme", next);
      } catch {
        // storage unavailable (private mode): the choice lasts for this page only
      }
      return next;
    });
  }, []);

  return { theme, toggle };
}

/** Resolved values of CSS custom properties, re-read whenever `key` (the theme) changes.
 *  Recharts draws SVG from props, so chart colors are read from the theme tokens here. */
export function useCssVars<T extends string>(names: readonly T[], key: string): Record<T, string> {
  const read = useCallback(() => {
    const style = getComputedStyle(document.documentElement);
    return Object.fromEntries(names.map((n) => [n, style.getPropertyValue(`--${n}`).trim()])) as Record<T, string>;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [names.join(",")]);
  const [values, setValues] = useState(read);
  useEffect(() => setValues(read()), [key, read]);
  return values;
}
