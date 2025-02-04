import React, { useRef, useState, useEffect } from 'react';

const Message = ({ role, content }) => (
  <div className={`message ${role === 'user' ? 'user' : 'assistant'}`}>
    <div className="message-content">{content}</div>
  </div>
);

const AgentSpeech = () => {
  const [isConnected, setIsConnected] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [messages, setMessages] = useState([]);
  const peerConnectionRef = useRef(null);
  const dataChannelRef = useRef(null);
  const audioElementRef = useRef(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const processResult = (data) => {
    if (data === null || data === undefined) {
      return null;
    }

    if (Array.isArray(data)) {
      return data.map(item => processResult(item));
    }

    if (typeof data === 'object') {
      const processed = {};
      for (const [key, value] of Object.entries(data)) {
        processed[key] = processResult(value);
      }
      return processed;
    }

    // Handle special string values that represent numbers
    if (data === "NaN") {
      return 0; // Or another appropriate default value
    }
    if (data === "Infinity") {
      return Number.MAX_SAFE_INTEGER;
    }
    if (data === "-Infinity") {
      return Number.MIN_SAFE_INTEGER;
    }

    return data;
  };

  const handleRealtimeEvent = async (event, dataChannel) => {
    console.log('Received event:', event);

    // Handle function calls
    if (event.type === 'response.function_call_arguments.done') {
      console.log('Function call completed:', event);
      const { arguments: args, call_id, name } = event;
      const parsedArgs = JSON.parse(args);

      try {
        // Call our backend to execute the function
        const response = await fetch('http://127.0.0.1:8000/api/v1/execute_function', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            function_name: name,
            arguments: parsedArgs,
            call_id: call_id
          })
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        console.log('Function execution result:', result);

        // Process the result and format for the model
        let formattedResult;
        try {
          // Attempt to format the result, handling special values
          formattedResult = processResult(result.data);
        } catch (formatError) {
          console.error('Error formatting result:', formatError);
          throw new Error('Failed to format function result');
        }

        // Send the processed result back to the model
        if (dataChannel && dataChannel.readyState === 'open') {
          const functionResult = {
            type: "conversation.item.create",
            item: {
              type: "function_call_output",
              call_id: call_id,
              output: JSON.stringify({
                status: 'success',
                data: formattedResult
              })
            }
          };

          console.log('Sending function result to model:', functionResult);
          dataChannel.send(JSON.stringify(functionResult));

          // Request model to continue
          const continueResponse = {
            type: "response.create"
          };
          dataChannel.send(JSON.stringify(continueResponse));
        }

      } catch (error) {
        console.error('Error executing function:', error);
        if (dataChannel && dataChannel.readyState === 'open') {
          const errorResult = {
            type: "conversation.item.create",
            item: {
              type: "function_call_output",
              call_id: call_id,
              output: JSON.stringify({
                status: 'error',
                error: error.message
              })
            }
          };
          dataChannel.send(JSON.stringify(errorResult));
        }
      }
      return;
    }

    // Handle other events
    switch (event.type) {
      case 'text.stream':
        setMessages(prev => {
          const lastMessage = prev[prev.length - 1];
          if (lastMessage?.role === 'assistant') {
            return [
              ...prev.slice(0, -1),
              { ...lastMessage, content: lastMessage.content + event.text }
            ];
          }
          return [...prev, { role: 'assistant', content: event.text }];
        });
        break;

      case 'audio.stream.end':
        setIsSpeaking(false);
        break;
    }
  };

  // Initialize WebRTC with function handling configuration
  const initializeWebRTC = async () => {
    try {
      console.log('Starting WebRTC initialization...');

      // 1. First get the session data from our backend
      const agentResponse = await fetch('http://127.0.0.1:8000/api/v1/query_agent', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          query: "Start session",
          context: {
            mode: 'realtime',
            function_handling: 'backend'
          }
        })
      });

      const agentData = await agentResponse.json();
      console.log('Agent response:', agentData);

      // Extract session from the correct location in the response
      const session = agentData.supporting_data?.session;
      if (!session?.client_secret?.value) {
        throw new Error('No valid session data received from agent');
      }

      const EPHEMERAL_KEY = session.client_secret.value;
      console.log('Got ephemeral key:', EPHEMERAL_KEY.substring(0, 10) + '...');

      // 2. Create and store the peer connection
      const peerConnection = new RTCPeerConnection();
      peerConnectionRef.current = peerConnection;

      // 3. Set up audio element
      const audioEl = document.createElement('audio');
      audioEl.autoplay = true;
      audioElementRef.current = audioEl;

      // 4. Handle incoming audio tracks
      peerConnection.ontrack = (e) => {
        console.log('Received audio track');
        audioEl.srcObject = e.streams[0];
        setIsSpeaking(true);
      };

      // 5. Add local audio track
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        peerConnection.addTrack(stream.getTracks()[0]);
      } catch (mediaError) {
        console.error('Media access error:', mediaError);
        throw new Error('Could not access microphone');
      }

      // 6. Create and configure data channel
      const dataChannel = peerConnection.createDataChannel('oai-events');
      dataChannelRef.current = dataChannel;

      dataChannel.onopen = () => {
        console.log('Data channel opened');
        setIsConnected(true);

        // Send initial session configuration
        const sessionConfig = {
          type: "session.update",
          session: {
            tools: session.tools || [],
            tool_choice: "auto",
            instructions: session.instructions,
            temperature: 0.7,
            max_response_output_tokens: "inf"
          }
        };
        console.log('Sending session config:', sessionConfig);
        dataChannel.send(JSON.stringify(sessionConfig));
      };

      dataChannel.onmessage = (e) => {
        const event = JSON.parse(e.data);
        handleRealtimeEvent(event, dataChannel);
      };

      dataChannel.onerror = (error) => {
        console.error('Data channel error:', error);
      };

      // 7. Create and set local description
      const offer = await peerConnection.createOffer();
      await peerConnection.setLocalDescription(offer);

      // 8. Get remote description from OpenAI
      const sdpResponse = await fetch(`https://api.openai.com/v1/realtime?model=${session.model}`, {
        method: 'POST',
        body: offer.sdp,
        headers: {
          Authorization: `Bearer ${EPHEMERAL_KEY}`,
          'Content-Type': 'application/sdp'
        },
      });

      if (!sdpResponse.ok) {
        const errorText = await sdpResponse.text();
        throw new Error(`SDP request failed: ${sdpResponse.status} - ${errorText}`);
      }

      // 9. Set remote description
      const answer = {
        type: 'answer',
        sdp: await sdpResponse.text()
      };

      await peerConnection.setRemoteDescription(answer);
      console.log('WebRTC initialization complete!');

    } catch (error) {
      console.error('Detailed initialization error:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Connection failed: ${error.message}`
      }]);

      // Clean up any partial initialization
      if (peerConnectionRef.current) {
        peerConnectionRef.current.close();
        peerConnectionRef.current = null;
      }
      if (dataChannelRef.current) {
        dataChannelRef.current.close();
        dataChannelRef.current = null;
      }
      setIsConnected(false);
      throw error;
    }
  };

  const startConversation = async () => {
    if (!isConnected) {
      await initializeWebRTC();
    }
    setIsListening(true);
  };

  const stopConversation = () => {
    if (peerConnectionRef.current) {
      peerConnectionRef.current.close();
    }
    if (dataChannelRef.current) {
      dataChannelRef.current.close();
    }
    setIsConnected(false);
    setIsListening(false);
    setIsSpeaking(false);
  };

  useEffect(() => {
    return () => {
      stopConversation();
    };
  }, []);

  // Function to send text message
  const sendMessage = async (text) => {
    if (!dataChannelRef.current) return;

    setMessages(prev => [...prev, { role: 'user', content: text }]);

    const messageEvent = {
      type: "conversation.item.create",
      item: {
        type: "message",
        role: "user",
        content: [{
          type: "input_text",
          text: text
        }]
      }
    };

    dataChannelRef.current.send(JSON.stringify(messageEvent));

    const responseEvent = {
      type: "response.create"
    };

    dataChannelRef.current.send(JSON.stringify(responseEvent));
  };

  return (
    <div className="app-wrapper">
      <div className="app-container">
        <div className="left-panel">
          <button
            className="action-button"
            onClick={isConnected ? stopConversation : startConversation}
          >
            {isListening ? 'Stop' : 'Start Conversation'}
          </button>
          {/* Added test button for sending a message */}
          {isConnected && (
            <button
              className="action-button mt-4"
              onClick={() => sendMessage("what's the best model by f1 score?")}
            >
              Test Query
            </button>
          )}
          <div
            className={`circle ${isListening ? 'listening' : ''} ${isSpeaking ? 'speaking' : ''}`}
          />
        </div>
        <div className="right-panel">
          <div className="conversation">
            {messages.map((msg, index) => (
              <Message key={index} role={msg.role} content={msg.content} />
            ))}
            <div ref={messagesEndRef} />
          </div>
        </div>
      </div>
      <style>{`
        .app-wrapper {
          min-height: 100vh;
          background: #f9f9f9;
          display: flex;
          justify-content: center;
        }

        .app-container {
          display: grid;
          grid-template-columns: minmax(300px, 1fr) minmax(600px, 2fr);
          width: 100%;
          max-width: 1600px;
        }

        .left-panel {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          background: #f9f9f9;
          padding: 2rem;
          position: sticky;
          top: 0;
          height: 100vh;
        }

        .right-panel {
          padding: 2rem;
          background: white;
          border-left: 1px solid #eee;
          min-height: 100vh;
          overflow-y: auto;
        }

        .conversation {
          display: flex;
          flex-direction: column;
          gap: 1rem;
        }

        .message {
          padding: 1rem;
          border-radius: 8px;
          max-width: 80%;
        }

        .message.user {
          background: #e3f2fd;
          align-self: flex-end;
        }

        .message.assistant {
          background: #f5f5f5;
          align-self: flex-start;
        }

        .action-button {
          background-color: #6200ee;
          color: #fff;
          border: none;
          padding: 0.75rem 1.5rem;
          border-radius: 4px;
          cursor: pointer;
          font-size: 1rem;
          margin-bottom: 1rem;
          transition: background-color 0.3s ease;
        }

        .action-button:hover {
          background-color: #3700b3;
        }

        .circle {
          width: 150px;
          height: 150px;
          border-radius: 50%;
          margin-top: 1rem;
          background: #ccc;
          transition: background 0.5s;
        }

        .listening {
          animation: pulse-blue 1.5s infinite;
          background: #2196f3;
        }

        .speaking {
          animation: pulse-green 1.5s infinite;
          background: #4caf50;
        }

        .mt-4 {
          margin-top: 1rem;
        }

        @keyframes pulse-blue {
          0% {
            transform: scale(1);
            box-shadow: 0 0 0 0 rgba(33, 150, 243, 0.4);
          }
          70% {
            transform: scale(1.1);
            box-shadow: 0 0 20px 30px rgba(33, 150, 243, 0);
          }
          100% {
            transform: scale(1);
            box-shadow: 0 0 0 0 rgba(33, 150, 243, 0);
          }
        }

        @keyframes pulse-green {
          0% {
            transform: scale(1);
            box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.4);
          }
          70% {
            transform: scale(1.1);
            box-shadow: 0 0 20px 30px rgba(76, 175, 80, 0);
          }
          100% {
            transform: scale(1);
            box-shadow: 0 0 0 0 rgba(76, 175, 80, 0);
          }
        }
      `}</style>
    </div>
  );
};

export default AgentSpeech;
