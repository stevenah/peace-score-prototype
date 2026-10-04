"use client";

import { useEffect, useRef, useState } from "react";
import {
  initialCoachState,
  nextCoachState,
  type CoachInputs,
  type CoachPrompt,
} from "@/lib/live/coaching";

/** Same prompt as far as the user can tell: same rule AND same wording. */
function samePrompt(a: CoachPrompt | null, b: CoachPrompt | null): boolean {
  return a?.id === b?.id && a?.text === b?.text;
}

/**
 * Drives the coaching state machine against a real clock.
 *
 * It re-evaluates on a timer as well as on input change, because clearing a
 * prompt is time-based: the condition going away is not enough on its own, it
 * has to stay away.
 */
export function useCoaching(inputs: CoachInputs): CoachPrompt | null {
  const stateRef = useRef(initialCoachState());
  const inputsRef = useRef(inputs);
  const [prompt, setPrompt] = useState<CoachPrompt | null>(null);

  useEffect(() => {
    inputsRef.current = inputs;
  }, [inputs]);

  useEffect(() => {
    const tick = () => {
      const next = nextCoachState(stateRef.current, inputsRef.current, Date.now());
      const changed = !samePrompt(next.active, stateRef.current.active);
      stateRef.current = next;
      if (changed) setPrompt(next.active);
    };
    tick();
    const id = setInterval(tick, 250);
    return () => clearInterval(id);
  }, []);

  // Re-evaluate immediately when the inputs change, so alerts are not delayed
  // by up to a tick.
  useEffect(() => {
    const next = nextCoachState(stateRef.current, inputs, Date.now());
    const changed = !samePrompt(next.active, stateRef.current.active);
    stateRef.current = next;
    if (changed) setPrompt(next.active);
  }, [inputs]);

  return prompt;
}
