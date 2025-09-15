import './globals.css';

export const metadata = {
  title: 'Computer Vision Demo',
  description: 'Upload an image to get AI predictions',
  viewport: 'width=device-width, initial-scale=1',
  icons: {
    icon: '/logo-white-fade.png',
    shortcut: '/logo-white-fade.png',
    apple: '/logo-white-fade.png',
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1" />
      </head>
      <body>{children}</body>
    </html>
  );
}
