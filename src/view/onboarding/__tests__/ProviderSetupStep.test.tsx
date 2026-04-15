import { render, screen, fireEvent, act } from '@testing-library/react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { ProviderSetupStep } from '../ProviderSetupStep';
import { invoke } from '../../../testUtils/mocks/tauri';

describe('ProviderSetupStep', () => {
  beforeEach(() => {
    invoke.mockClear();
  });

  it('renders provider setup title', () => {
    render(<ProviderSetupStep initialProvider="gemini" onComplete={vi.fn()} />);
    expect(screen.getByText('Connect your AI provider')).toBeInTheDocument();
  });

  it('requires API key for gemini provider', async () => {
    render(<ProviderSetupStep initialProvider="gemini" onComplete={vi.fn()} />);

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /save and continue/i }));
    });

    expect(screen.getByText(/please enter your api key/i)).toBeInTheDocument();
  });

  it('saves setup and completes', async () => {
    const onComplete = vi.fn();
    invoke.mockResolvedValue(undefined);

    render(<ProviderSetupStep initialProvider="gemini" onComplete={onComplete} />);

    await act(async () => {
      fireEvent.change(screen.getByLabelText('API key'), {
        target: { value: 'AIza-test' },
      });
      fireEvent.click(screen.getByRole('button', { name: /save and continue/i }));
    });

    expect(invoke).toHaveBeenCalledWith('save_provider_setup', {
      payload: {
        provider: 'gemini',
        apiKey: 'AIza-test',
        model: 'gemini-2.5-flash',
      },
    });
    expect(onComplete).toHaveBeenCalledTimes(1);
  });

  it('saves Perplexity provider setup', async () => {
    const onComplete = vi.fn();
    invoke.mockResolvedValue(undefined);

    render(<ProviderSetupStep initialProvider="perplexity" onComplete={onComplete} />);

    await act(async () => {
      fireEvent.change(screen.getByLabelText('API key'), {
        target: { value: 'pplx-test' },
      });
      fireEvent.click(screen.getByRole('button', { name: /save and continue/i }));
    });

    expect(invoke).toHaveBeenCalledWith('save_provider_setup', {
      payload: {
        provider: 'perplexity',
        apiKey: 'pplx-test',
        model: 'sonar',
      },
    });
    expect(onComplete).toHaveBeenCalledTimes(1);
  });
});
