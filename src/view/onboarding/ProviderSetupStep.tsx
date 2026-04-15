import { useMemo, useState } from 'react';
import type React from 'react';
import { invoke } from '@tauri-apps/api/core';

type Provider = 'gemini' | 'perplexity' | 'openai' | 'anthropic' | 'ollama';

const DEFAULT_MODEL_BY_PROVIDER: Record<Provider, string> = {
  gemini: 'gemini-2.5-flash',
  perplexity: 'xai/grok-4-1-fast-non-reasoning',
  openai: 'gpt-4.1-mini',
  anthropic: 'claude-3-7-sonnet-latest',
  ollama: 'gemma4:e2b',
};

interface Props {
  initialProvider: string;
  onComplete: () => void;
}

/**
 * First-run provider/API-key setup shown before normal overlay interaction.
 */
export function ProviderSetupStep({ initialProvider, onComplete }: Props) {
  const normalizedInitial = useMemo<Provider>(() => {
    const v = initialProvider.toLowerCase();
    if (
      v === 'openai' ||
      v === 'anthropic' ||
      v === 'ollama' ||
      v === 'gemini' ||
      v === 'perplexity'
    ) {
      return v;
    }
    return 'gemini';
  }, [initialProvider]);

  const [provider, setProvider] = useState<Provider>(normalizedInitial);
  const [apiKey, setApiKey] = useState('');
  const [model, setModel] = useState(DEFAULT_MODEL_BY_PROVIDER[normalizedInitial]);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const requiresApiKey = provider !== 'ollama';

  const onProviderChange = (nextProvider: Provider) => {
    setProvider(nextProvider);
    setError(null);
    setModel((prev) => prev.trim() || DEFAULT_MODEL_BY_PROVIDER[nextProvider]);
  };

  const handleSave = async () => {
    const trimmedModel = model.trim() || DEFAULT_MODEL_BY_PROVIDER[provider];
    const trimmedApiKey = apiKey.trim();
    if (requiresApiKey && !trimmedApiKey) {
      setError('Please enter your API key to continue.');
      return;
    }

    setIsSaving(true);
    setError(null);
    try {
      await invoke('save_provider_setup', {
        payload: {
          provider,
          apiKey: trimmedApiKey,
          model: trimmedModel,
        },
      });
      onComplete();
    } catch {
      setError('Could not save settings. Please try again.');
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div
      style={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'transparent',
        fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
      }}
    >
      <div
        style={{
          width: 460,
          background:
            'radial-gradient(ellipse 80% 55% at 50% 0%, rgba(255,141,92,0.14) 0%, rgba(28,24,20,0.97) 60%), rgba(28,24,20,0.97)',
          border: '1px solid rgba(255, 141, 92, 0.2)',
          borderRadius: 24,
          padding: '28px 24px',
          boxShadow: '0 0 40px rgba(255,100,40,0.07)',
          color: '#f0f0f2',
        }}
      >
        <h1 style={{ margin: 0, fontSize: 22, lineHeight: 1.2 }}>Connect your AI provider</h1>
        <p style={{ margin: '8px 0 20px', color: 'rgba(255,255,255,0.55)', fontSize: 13 }}>
          One-time setup. You can change this later in settings.
        </p>

        <label style={labelStyle}>Provider</label>
        <select
          aria-label="Provider"
          value={provider}
          onChange={(e) => onProviderChange(e.target.value as Provider)}
          style={inputStyle}
        >
          <option value="gemini">Gemini</option>
          <option value="perplexity">Perplexity</option>
          <option value="openai">OpenAI</option>
          <option value="anthropic">Anthropic</option>
          <option value="ollama">Ollama (local)</option>
        </select>

        {requiresApiKey ? (
          <>
            <label style={labelStyle}>API key</label>
            <input
              aria-label="API key"
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder={
                provider === 'gemini'
                  ? 'AIza...'
                  : provider === 'perplexity'
                    ? 'pplx-...'
                    : 'Paste your API key'
              }
              style={inputStyle}
            />
          </>
        ) : null}

        <label style={labelStyle}>Model</label>
        <input
          aria-label="Model"
          type="text"
          value={model}
          onChange={(e) => setModel(e.target.value)}
          placeholder={DEFAULT_MODEL_BY_PROVIDER[provider]}
          style={inputStyle}
        />

        {error ? (
          <p style={{ margin: '10px 2px 0', color: '#ff8a8a', fontSize: 12 }}>{error}</p>
        ) : null}

        <button
          onClick={() => void handleSave()}
          disabled={isSaving}
          style={{
            marginTop: 16,
            width: '100%',
            padding: '11px 12px',
            borderRadius: 12,
            border: 'none',
            cursor: isSaving ? 'not-allowed' : 'pointer',
            background: 'linear-gradient(135deg, #ff8d5c 0%, #d45a1e 100%)',
            color: '#fff',
            fontSize: 14,
            fontWeight: 600,
            opacity: isSaving ? 0.7 : 1,
          }}
        >
          {isSaving ? 'Saving…' : 'Save and continue'}
        </button>
      </div>
    </div>
  );
}

const labelStyle: React.CSSProperties = {
  display: 'block',
  marginTop: 10,
  marginBottom: 6,
  fontSize: 12,
  color: 'rgba(255,255,255,0.7)',
};

const inputStyle: React.CSSProperties = {
  width: '100%',
  boxSizing: 'border-box',
  borderRadius: 10,
  border: '1px solid rgba(255,255,255,0.12)',
  background: 'rgba(255,255,255,0.04)',
  color: '#fff',
  padding: '10px 12px',
  outline: 'none',
  fontSize: 13,
};
