import React from 'react';
import { Button } from '@/components/ui/button';
import { useToast } from '@/components/ui/use-toast';

interface FeedbackProps {
  tradeId: string;
}

export const Feedback: React.FC<FeedbackProps> = ({ tradeId }) => {
  const { toast } = useToast();

  const submitFeedback = async (feedback: 'good' | 'bad') => {
    try {
      const response = await fetch('http://localhost:3006/api/feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ tradeId, feedback }),
      });

      if (response.ok) {
        toast({
          title: 'Feedback Submitted',
          description: `You marked trade ${tradeId} as a ${feedback} trade.`,
        });
      } else {
        toast({
          title: 'Error',
          description: 'Failed to submit feedback.',
          variant: 'destructive',
        });
      }
    } catch (error) {
      console.error('Feedback submission error:', error);
      toast({
        title: 'Error',
        description: 'An error occurred while submitting feedback.',
        variant: 'destructive',
      });
    }
  };

  return (
    <div className="flex space-x-2">
      <Button onClick={() => submitFeedback('good')} variant="outline" size="sm">
        Good Trade
      </Button>
      <Button onClick={() => submitFeedback('bad')} variant="destructive" size="sm">
        Bad Trade
      </Button>
    </div>
  );
};
