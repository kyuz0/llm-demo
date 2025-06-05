# app.py
from flask import Flask, request, jsonify, render_template
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import math
import numpy as np

app = Flask(__name__)

# Model info
MODEL_ID = "google/gemma-2-2b"  # Upgraded to a more capable small model

# Initialize model and tokenizer
model = None
tokenizer = None

def load_model_if_needed():
    global model, tokenizer
    if model is None or tokenizer is None:
        # Setup 4-bit quantization
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)

        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=quantization_config,
        )
        
        # If tokenizer doesn't have pad token, set it
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

@app.route('/')
def index():
    load_model_if_needed()
    return render_template('index.html')

@app.route('/api/generate', methods=['POST'])
def generate():
    load_model_if_needed()
    data = request.json
    prompt = data.get('prompt', '')
    max_tokens = int(data.get('max_tokens', 50))
    temperature = float(data.get('temperature', 0.7))
    
    # For alternative token selection and continuation
    prefix_tokens = data.get('prefix_tokens', [])
    selected_token = data.get('selected_token', None)
    
    # If we have token selection, add it to the prefix
    if selected_token:
        full_prompt = prompt
        if prefix_tokens:
            full_prompt += ''.join(prefix_tokens)
        full_prompt += selected_token
        prompt = full_prompt
    
    try:
        # Tokenize input
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            input_ids = input_ids.to("cuda")
        
        # Generate with logprobs
        with torch.no_grad():
            # Forward pass to get logits
            outputs = model(input_ids)
            logits = outputs.logits
            
            # Only use the last token's logits for generation
            next_token_logits = logits[0, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Get probabilities
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            
            # Generate one token at a time to capture logprobs for each
            generated_tokens = []
            token_probs = []
            token_ids = []
            
            for _ in range(max_tokens):
                # Sample from the distribution
                if temperature == 0:
                    next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                else:
                    next_token_id = torch.multinomial(probs, num_samples=1)
                
                # Get top k alternatives for this token
                topk = 10
                topk_probs, topk_indices = torch.topk(probs, k=topk)
                
                # Save token and its alternatives
                token_ids.append(next_token_id.item())
                token_probs.append({
                    'token_id': next_token_id.item(),
                    'token': tokenizer.decode(next_token_id),
                    'logprob': torch.log(probs[next_token_id]).item(),
                    'alternatives': [
                        {
                            'token_id': idx.item(),
                            'token': tokenizer.decode(idx),
                            'logprob': torch.log(probs[idx]).item(),
                            'probability': probs[idx].item() * 100
                        }
                        for idx in topk_indices
                    ]
                })
                
                # Append to input and repeat
                input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)
                
                # Get next token distribution
                with torch.no_grad():
                    outputs = model(input_ids)
                    logits = outputs.logits
                    next_token_logits = logits[0, -1, :]
                    if temperature != 1.0:
                        next_token_logits = next_token_logits / temperature
                    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                
                # Save generated token
                generated_tokens.append(tokenizer.decode(next_token_id))
        
        # Process the results
        processed_output = process_generation_output(generated_tokens, token_probs)
        
        # If this was a continuation from token selection, add metadata
        if selected_token:
            processed_output['continuationInfo'] = {
                'originalPrompt': data.get('original_prompt', prompt),
                'prefixTokens': prefix_tokens,
                'selectedToken': selected_token
            }
            
        return jsonify(processed_output)
    except Exception as e:
        error_msg = str(e)
        print(f"Error generating completion: {error_msg}")
        return jsonify({"error": f"Error: {error_msg}"}), 500

def process_generation_output(generated_tokens, token_probs):
    tokens = []
    
    for i, (token_text, token_prob) in enumerate(zip(generated_tokens, token_probs)):
        # Clean up token text (sometimes has extra spaces or control chars)
        clean_text = token_text.replace('▁', ' ')
        
        token_data = {
            'token': i,
            'text': clean_text,
            'logprob': token_prob['logprob'],
            'logprobs': []
        }
        
        # Add alternatives
        for alt in token_prob['alternatives']:
            clean_alt_text = alt['token'].replace('▁', ' ')
            token_data['logprobs'].append({
                'token': clean_alt_text,
                'logprob': alt['logprob'],
                'probability': alt['probability']
            })
        
        tokens.append(token_data)
    
    return {
        'text': ''.join(generated_tokens),
        'tokens': tokens
    }

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    
    with open('templates/index.html', 'w') as f:
        f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Base LLM Demo</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }
        h1 {
            text-align: center;
            margin-bottom: 30px;
            color: #2c3e50;
        }
        .container {
            display: flex;
            flex-direction: column;
            gap: 30px;
        }
        .input-section, .output-section {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .controls {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-bottom: 20px;
        }
        @media (max-width: 768px) {
            .controls {
                grid-template-columns: 1fr;
            }
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        textarea {
            width: 100%;
            min-height: 100px;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
            font-size: 16px;
            margin-bottom: 10px;
        }
        input[type="number"], input[type="range"] {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 16px;
        }
        button {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 4px;
            font-size: 16px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        button:hover {
            background-color: #2980b9;
        }
        .output-display {
            min-height: 100px;
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 15px;
            white-space: pre-wrap;
            overflow-wrap: break-word;
        }
        .token {
            display: inline-block;
            padding: 2px 4px;
            border-radius: 3px;
            margin: 0 2px;
            cursor: pointer;
            position: relative;
            background-color: rgba(52, 152, 219, 0.1);
            transition: background-color 0.2s;
        }
        .token:hover {
            background-color: rgba(52, 152, 219, 0.3);
        }
        .token-info {
            display: none;
            position: absolute;
            bottom: 100%;
            left: 0;
            background-color: #2c3e50;
            color: white;
            padding: 10px;
            border-radius: 4px;
            font-size: 14px;
            min-width: 300px;
            z-index: 10;
            box-shadow: 0 3px 10px rgba(0,0,0,0.2);
        }
        .token:hover .token-info {
            display: block;
        }
        .loading {
            text-align: center;
            padding: 20px;
            font-style: italic;
            color: #7f8c8d;
        }
        .slider-container {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .slider-value {
            min-width: 40px;
            text-align: center;
        }
        .explanation {
            margin-top: 30px;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }
        .probability-display {
            display: flex;
            align-items: center;
            margin-top: 4px;
        }
        .probability-bar {
            flex-grow: 1;
            height: 8px;
            background-color: #eee;
            border-radius: 4px;
            position: relative;
            overflow: hidden;
            margin: 0 10px;
        }
        .probability-fill {
            position: absolute;
            top: 0;
            left: 0;
            height: 100%;
            background-color: #3498db;
        }
        .probability-value {
            min-width: 60px;
            text-align: right;
            font-size: 12px;
        }
        .alt-token {
            cursor: pointer;
            padding: 2px 6px;
            margin-top: 4px;
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            transition: background-color 0.2s;
        }
        .alt-token:hover {
            background-color: rgba(255, 255, 255, 0.2);
        }
        .token.highlighted {
            background-color: rgba(46, 204, 113, 0.3);
        }
        .alt-token.selected {
            background-color: rgba(46, 204, 113, 0.3);
        }
    </style>
</head>
<body>
    <h1>Base LLM Demo - Google Gemma2-2B (base, 4bit)</h1>
    
    <div class="container">
        <div class="input-section">
            <h2>Input</h2>
            <label for="prompt">Enter your prompt:</label>
            <textarea id="prompt" placeholder="Once upon a time..."></textarea>
            
            <div class="controls">
                <div>
                    <label for="max-tokens">Max Tokens:</label>
                    <input type="number" id="max-tokens" min="1" max="200" value="50">
                </div>
                <div style="margin-left: 15px">
                    <label for="temperature">Temperature:</label>
                    <div class="slider-container">
                        <input type="range" id="temperature" min="0" max="2" step="0.1" value="0.7">
                        <span class="slider-value" id="temp-value">0.7</span>
                    </div>
                </div>
            </div>
            
            <button id="generate-btn">Generate</button>
        </div>
        
        <div class="output-section">
            <h2>Output</h2>
            <div id="loading" class="loading" style="display: none;">Generating...</div>
            <div id="output" class="output-display"></div>
        </div>
        
        <div class="explanation">
            <h2>About This Demo</h2>
            <p>This demo uses Google Gemma2-2B: a small, base language model without instruction tuning.</p>
            <p>When you hover over tokens, you can see the alternatives the model considered. Click any alternative to regenerate from that point.</p>
        </div>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const promptInput = document.getElementById('prompt');
            const maxTokensInput = document.getElementById('max-tokens');
            const temperatureSlider = document.getElementById('temperature');
            const tempValue = document.getElementById('temp-value');
            const generateBtn = document.getElementById('generate-btn');
            const outputDiv = document.getElementById('output');
            const loadingDiv = document.getElementById('loading');
            
            let originalPrompt = '';
            let generatedTokens = [];
            let selectedTokenIndex = -1;
            let selectedAlternativeToken = '';
            
            temperatureSlider.addEventListener('input', function() {
                tempValue.textContent = this.value;
            });
            
            generateBtn.addEventListener('click', function() {
                originalPrompt = promptInput.value.trim();
                generatedTokens = [];
                selectedTokenIndex = -1;
                selectedAlternativeToken = '';
                
                if (!originalPrompt) {
                    alert('Please enter a prompt');
                    return;
                }
                
                outputDiv.innerHTML = '';
                loadingDiv.style.display = 'block';
                
                generateText(originalPrompt);
            });
            
            function generateText(prompt, prefixTokens = [], selectedToken = null) {
                fetch('/api/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        prompt: prompt,
                        original_prompt: originalPrompt,
                        max_tokens: maxTokensInput.value,
                        temperature: temperatureSlider.value,
                        prefix_tokens: prefixTokens,
                        selected_token: selectedToken
                    })
                })
                .then(response => response.json())
                .then(data => {
                    loadingDiv.style.display = 'none';
                    
                    if (data.error) {
                        outputDiv.textContent = data.error;
                        return;
                    }
                    
                    if (prefixTokens.length === 0 && !selectedToken) {
                        generatedTokens = data.tokens.map(t => t.text);
                    } else if (selectedToken) {
                        generatedTokens = prefixTokens.concat([selectedToken]);
                        data.tokens.forEach(token => {
                            generatedTokens.push(token.text);
                        });
                    }
                    
                    displayTokens(data.tokens, prefixTokens, selectedToken);
                })
                .catch(error => {
                    loadingDiv.style.display = 'none';
                    outputDiv.textContent = 'Error: ' + error.message;
                    console.error('Error:', error);
                });
            }
            
            function displayTokens(tokens, prefixTokens = [], selectedToken = null) {
                outputDiv.innerHTML = '';
                
                if (prefixTokens.length > 0) {
                    for (let i = 0; i < prefixTokens.length; i++) {
                        const tokenSpan = document.createElement('span');
                        tokenSpan.className = 'token';
                        tokenSpan.textContent = prefixTokens[i];
                        outputDiv.appendChild(tokenSpan);
                    }
                }
                
                if (selectedToken) {
                    const tokenSpan = document.createElement('span');
                    tokenSpan.className = 'token highlighted';
                    tokenSpan.textContent = selectedToken;
                    outputDiv.appendChild(tokenSpan);
                }
                
                tokens.forEach((tokenData, index) => {
                    const tokenSpan = document.createElement('span');
                    tokenSpan.className = 'token';
                    tokenSpan.textContent = tokenData.text;
                    
                    const tokenInfo = document.createElement('div');
                    tokenInfo.className = 'token-info';
                    
                    if (tokenData.logprobs && tokenData.logprobs.length > 0) {
                        const tokenInfoTitle = document.createElement('div');
                        tokenInfoTitle.innerHTML = `<strong>Token:</strong> "${tokenData.text}"`;
                        tokenInfo.appendChild(tokenInfoTitle);
                        
                        const tokenInfoSubtitle = document.createElement('div');
                        tokenInfoSubtitle.innerHTML = '<strong>Alternatives:</strong>';
                        tokenInfo.appendChild(tokenInfoSubtitle);
                        
                        const currentTokenIndex = (prefixTokens.length + (selectedToken ? 1 : 0) + index);
                        
                        tokenData.logprobs.forEach(prob => {
                            const probDiv = document.createElement('div');
                            probDiv.className = 'alt-token';
                            probDiv.style.marginTop = '8px';
                            
                            if (selectedToken && prob.token === selectedToken && index === 0) {
                                probDiv.classList.add('selected');
                            }
                            
                            probDiv.onclick = function() {
                                selectedTokenIndex = currentTokenIndex;
                                selectedAlternativeToken = prob.token;
                                
                                if (confirm(`Regenerate from token "${prob.token}" (${prob.probability.toFixed(2)}% probability)?`)) {
                                    const prefixTokensForRegeneration = index === 0 && !prefixTokens.length 
                                        ? [] 
                                        : generatedTokens.slice(0, currentTokenIndex);
                                    
                                    outputDiv.innerHTML = '';
                                    loadingDiv.style.display = 'block';
                                    
                                    generateText(originalPrompt, prefixTokensForRegeneration, prob.token);
                                }
                            };
                            
                            const probText = document.createElement('div');
                            probText.textContent = `"${prob.token}"`;
                            probDiv.appendChild(probText);
                            
                            const probabilityDisplay = document.createElement('div');
                            probabilityDisplay.className = 'probability-display';
                            
                            const probabilityBar = document.createElement('div');
                            probabilityBar.className = 'probability-bar';
                            
                            const probabilityFill = document.createElement('div');
                            probabilityFill.className = 'probability-fill';
                            probabilityFill.style.width = `${prob.probability}%`;
                            
                            const probabilityValue = document.createElement('div');
                            probabilityValue.className = 'probability-value';
                            probabilityValue.textContent = `${prob.probability.toFixed(2)}%`;
                            
                            probabilityBar.appendChild(probabilityFill);
                            probabilityDisplay.appendChild(probabilityBar);
                            probabilityDisplay.appendChild(probabilityValue);
                            
                            probDiv.appendChild(probabilityDisplay);
                            
                            tokenInfo.appendChild(probDiv);
                        });
                    } else {
                        tokenInfo.textContent = 'No probability data available';
                    }
                    
                    tokenSpan.appendChild(tokenInfo);
                    outputDiv.appendChild(tokenSpan);
                });
            }
        });
    </script>
</body>
</html>""")
    
    print("Starting LLM Demo with Google Gemma2-2B...")
    print("Visit http://localhost:5000 to use the app")
    app.run(debug=True)