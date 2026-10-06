import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import App from "./App";
import { pingBackend } from "./api/client";
import "./styles/theme.css";

// Start waking a sleeping backend right away, before React renders and asks for the samples.
pingBackend();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
