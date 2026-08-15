import { Navbar } from '@/components/Navbar'
import { Footer } from '@/components/Footer'
import { NotFoundPage } from '@/components/notfound/NotFoundPage'

export default function NotFoundApp() {
  return (
    <div className="flex min-h-screen flex-col bg-background">
      <Navbar />
      <main className="flex-1">
        <NotFoundPage />
      </main>
      <Footer />
    </div>
  )
}
