import { backendUrl, forwardResponse } from "@/lib/backend";

export async function POST() {
  try {
    const upstream = await fetch(backendUrl("/demo"), { method: "POST" });
    return forwardResponse(upstream);
  } catch {
    return Response.json(
      { detail: "Registration engine is waking up or unavailable. Try again in a moment." },
      { status: 503 },
    );
  }
}
