import { useState } from "react";
import ReactMarkdown from "react-markdown";
import { Lottie } from "lottie-react";

import doctorAnimation from "./assets/doctor-animation.json";

import "./App.css";


export default function App() {

  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content: "Hello! How can I help you today?",
      sources: []
    }
  ]);

  const [input, setInput] = useState("");

  const [loading, setLoading] = useState(false);

  const [error, setError] = useState("");


  async function sendMessage(event) {

    event.preventDefault();

    const text = input.trim();

    if (!text || loading) {
      return;
    }


    const userMessage = {
      role: "user",
      content: text
    };


    const updatedMessages = [
      ...messages,
      userMessage
    ];


    setMessages(updatedMessages);

    setInput("");

    setLoading(true);

    setError("");


    try {

      const response = await fetch(
        "/api/chat",
        {
          method: "POST",

          headers: {
            "Content-Type": "application/json"
          },

          body: JSON.stringify({
            messages: updatedMessages.map(
              (message) => ({
                role: message.role,
                content: message.content
              })
            )
          })
        }
      );


      const data = await response.json();


      if (!response.ok) {
        throw new Error(
          data.detail || "Failed to get response."
        );
      }


      const assistantMessage = {
        role: "assistant",
        content: data.answer,
        sources: data.sources || []
      };


      setMessages([
        ...updatedMessages,
        assistantMessage
      ]);

    } catch (error) {

      console.error(error);

      setError(
        "Something went wrong while generating the response."
      );

    } finally {

      setLoading(false);

    }
  }


  function clearChat() {

    setMessages([
      {
        role: "assistant",
        content: "Hello! How can I help you today?",
        sources: []
      }
    ]);

    setError("");
  }


  function handleKeyDown(event) {

    if (
      event.key === "Enter" &&
      !event.shiftKey
    ) {

      event.preventDefault();

      event.currentTarget.form.requestSubmit();
    }
  }


  return (

    <div className="app">

      <div className="chat-container">

        <header className="chat-header">

          <div className="header-title">

            <div className="doctor-avatar">

              <Lottie
                src={doctorAnimation}
                loop
                autoplay
              />

            </div>


            <div className="header-text">

              <h1>
                AI Assistant
              </h1>

              <p>
                MedQuAD-powered medical information assistant
              </p>

            </div>

          </div>


          <button
            className="clear-button"
            onClick={clearChat}
          >
            Clear
          </button>

        </header>


        <main className="messages">

          {messages.map(
            (message, index) => (

              <div
                key={index}
                className={`message-row ${message.role}`}
              >

                <div className="message">

                  <span className="message-author">

                    {
                      message.role === "user"
                        ? "You"
                        : "Assistant"
                    }

                  </span>


                  <ReactMarkdown>
                    {message.content}
                  </ReactMarkdown>


                  {
                    message.role === "assistant" &&
                    message.sources?.length > 0 && (

                      <div className="message-sources">

                        <strong>
                          Sources:
                        </strong>


                        {
                          message.sources.map(
                            (source, sourceIndex) => (

                              <a
                                key={sourceIndex}
                                href={source.url}
                                target="_blank"
                                rel="noopener noreferrer"
                              >
                                {source.name}
                              </a>

                            )
                          )
                        }

                      </div>

                    )
                  }

                </div>

              </div>

            )
          )}


          {loading && (

            <div className="message-row assistant">

              <div className="message">

                <span className="message-author">
                  Assistant
                </span>

                <p className="typing">
                  Thinking...
                </p>

              </div>

            </div>

          )}

        </main>


        {error && (

          <div className="error">
            {error}
          </div>

        )}


        <form
          className="input-area"
          onSubmit={sendMessage}
        >

          <textarea
            value={input}
            onChange={
              (event) =>
                setInput(event.target.value)
            }
            onKeyDown={handleKeyDown}
            placeholder="Ask a medical information question..."
            rows="2"
            autoFocus
          />


          <button
            type="submit"
            disabled={
              loading ||
              !input.trim()
            }
          >
            Send
          </button>

        </form>

      </div>

    </div>

  );
}