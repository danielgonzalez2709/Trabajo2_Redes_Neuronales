/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "export", // sitio estático: sin servidor
  reactStrictMode: true,
  allowedDevOrigins: ["*.trycloudflare.com"], // solo `next dev`: permite abrir la app por el túnel
};

export default nextConfig;
