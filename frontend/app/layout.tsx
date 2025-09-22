import './globals.css';

export const metadata = {
  title: 'Mimetic Demo',
  description: 'Test our Computer Vision Models',
  viewport: 'width=device-width, initial-scale=1',
  icons: {
    icon: '/logo-white-fade.png',
    shortcut: '/logo-white-fade.png',
    apple: '/logo-white-fade.png',
  },
  // Dark theme meta tags for mobile browsers
  themeColor: '#0f172a', // Dark slate background to match your design
  colorScheme: 'dark',
  other: {
    // Additional meta tags for mobile browser theming
    'msapplication-navbutton-color': '#0f172a',
    'apple-mobile-web-app-status-bar-style': 'black-translucent',
    'apple-mobile-web-app-capable': 'yes',
  }
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1" />
        {/* Additional meta tags for mobile browser theming */}
        <meta name="theme-color" content="#0f172a" />
        <meta name="msapplication-navbutton-color" content="#0f172a" />
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent" />
        <meta name="apple-mobile-web-app-capable" content="yes" />
        <style dangerouslySetInnerHTML={{
          __html: `
            /* Prevent white flash on page load */
            html, body { 
              background-color: #0f172a !important; 
              color-scheme: dark;
            }
            /* Ensure consistent dark scrollbar on webkit browsers */
            ::-webkit-scrollbar {
              background-color: #0f172a;
            }
            ::-webkit-scrollbar-track {
              background-color: #1e293b;
            }
            ::-webkit-scrollbar-thumb {
              background-color: #475569;
              border-radius: 4px;
            }
          `
        }} />
      </head>
      <body className="bg-[#0f172a] text-slate-200">{children}</body>
    </html>
  );
}
