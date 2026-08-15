import { Button } from '@/components/ui/button'

export function NotFoundPage() {
  return (
    <div className="mx-auto max-w-2xl px-6 py-24 text-center">
      <p className="text-sm font-semibold text-primary">404</p>
      <h1 className="mt-2 text-3xl font-bold">Page not found</h1>
      <p className="mt-4 text-muted-foreground">
        The page you're looking for doesn't exist or may have moved.
      </p>
      <div className="mt-8 flex flex-wrap justify-center gap-3">
        <Button asChild size="lg">
          <a href="/">Back to Home</a>
        </Button>
        <Button asChild size="lg" variant="outline">
          <a href="/search">Search Books</a>
        </Button>
      </div>
    </div>
  )
}
