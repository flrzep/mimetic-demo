import './globals.css';

export const metadata = {
  title: 'Computer Vision Demo',
  description: 'Upload an image to get AI predictions',
  icons: {
    icon: '/logo-white-fade.png',
    shortcut: '/logo-white-fade.png',
    apple: '/logo-white-fade.png',
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
