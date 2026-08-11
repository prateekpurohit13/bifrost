package gemini

import (
	"encoding/json"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
)

func TestServerSideToolCallStreaming(t *testing.T) {
	chunks := []string{
		`{"candidates":[{"index":0,"content":{"role":"model","parts":[{"thoughtSignature":"c2ln","toolCall":{"toolType":"GOOGLE_SEARCH_WEB","args":{"queries":["f1 winner 2026","f1 results"]},"id":"nqh2j2zy"}}]}}],"modelVersion":"gemini-3.5-flash"}`,
		`{"candidates":[{"index":0,"content":{"role":"model","parts":[{"thoughtSignature":"c2ln","toolResponse":{"toolType":"GOOGLE_SEARCH_WEB","response":{"search_suggestions":"<div/>"},"id":"nqh2j2zy"}}]}}],"modelVersion":"gemini-3.5-flash"}`,
		`{"candidates":[{"index":0,"content":{"role":"model","parts":[{"text":"Lando Norris won."}]}}],"modelVersion":"gemini-3.5-flash"}`,
		`{"candidates":[{"index":0,"finishReason":"STOP","groundingMetadata":{"webSearchQueries":["f1 winner 2026"],"groundingChunks":[{"web":{"uri":"https://x/1","title":"formula1.com"}}]}}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15},"modelVersion":"gemini-3.5-flash"}`,
	}
	state := acquireGeminiResponsesStreamState()
	defer releaseGeminiResponsesStreamState(state)
	seq := 0
	webSearchItems := map[string]*schemas.ResponsesWebSearchToolCallAction{}
	counts := map[string]int{}
	for _, c := range chunks {
		var resp GenerateContentResponse
		if err := json.Unmarshal([]byte(c), &resp); err != nil {
			t.Fatal(err)
		}
		events, err := resp.ToBifrostResponsesStream(seq, state)
		if err != nil {
			t.Fatal(err)
		}
		for _, e := range events {
			counts[string(e.Type)]++
			if e.Item != nil && e.Item.Type != nil && *e.Item.Type == schemas.ResponsesMessageTypeWebSearchCall {
				if e.Item.ResponsesToolMessage != nil && e.Item.ResponsesToolMessage.Action != nil {
					if a := e.Item.ResponsesToolMessage.Action.ResponsesWebSearchToolCallAction; a != nil && len(a.Queries) > 0 {
						webSearchItems[*e.Item.ID] = a
					}
				}
			}
		}
		seq += len(events)
	}
	t.Logf("web_search_call items: %d", len(webSearchItems))
	for id, a := range webSearchItems {
		j, _ := json.Marshal(map[string]any{"queries": a.Queries, "sources": len(a.Sources)})
		t.Logf("  id=%s %s", id, string(j))
	}
	t.Logf("reasoning items: %d", counts["response.output_item.added"])
	if len(webSearchItems) != 1 {
		t.Fatalf("expected exactly 1 web_search_call, got %d", len(webSearchItems))
	}
	if _, ok := webSearchItems["nqh2j2zy"]; !ok {
		t.Errorf("web_search_call did not use the model's own tool call id")
	}
	a := webSearchItems["nqh2j2zy"]
	if len(a.Queries) != 2 {
		t.Errorf("expected the toolCall's 2 queries, got %v", a.Queries)
	}
	if len(a.Sources) != 1 {
		t.Errorf("expected grounding sources merged in, got %d", len(a.Sources))
	}
}
