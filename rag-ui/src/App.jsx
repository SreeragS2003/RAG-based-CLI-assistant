import { useState } from "react";

function App() {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState([]);

  const sendMessage = async () => {
    if (!query.trim()) return;

    // Add user message
    setMessages((prev) => [...prev, { role: "user", text: query }]);

    // Add empty assistant message
    setMessages((prev) => [...prev, { role: "assistant", text: "" }]);

    const response = await fetch("http://127.0.0.1:8000/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        query: query,
        user_id: "user1",
      }),
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");

    let done = false;
    let accumulatedText = "";

    while (!done) {
      const { value, done: doneReading } = await reader.read();
      done = doneReading;

      const chunk = decoder.decode(value || new Uint8Array());
      accumulatedText += chunk;

      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: "assistant",
          text: accumulatedText,
        };
        return updated;
      });
    }

    setQuery("");
  };

  return (
    <div style={{ maxWidth: "600px", margin: "auto", padding: "20px" }}>
      <h2>Chat with RAG-based AI Agent</h2>

      <div style={{ minHeight: "300px", marginBottom: "20px" }}>
        {messages.map((msg, i) => (
          <div key={i} style={{ margin: "10px 0" }}>
            <strong>{msg.role === "user" ? "You" : "AI"}:</strong>{" "}
            {msg.text}
          </div>
        ))}
      </div>

      <input
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Ask something..."
        style={{ width: "80%", padding: "10px" }}
      />

      <button onClick={sendMessage} style={{ padding: "10px" , marginLeft: "10px"}}>
        Send
      </button>
    </div>
  );
}

export default App;