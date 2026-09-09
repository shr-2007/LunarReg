import { backendUrl, forwardResponse } from "@/lib/backend";

export async function POST(request: Request) {
  try {
    const form = await request.formData();
    const upstream = await fetch(backendUrl("/register"), { method: "POST", body: form });
    return forwardResponse(upstream);
  } catch {
    return Response.json(
      { detail: "Registration engine is unavailable. Start the FastAPI service or check the Render Blueprint." },
      { status: 503 },
    );
  }
}
