import collections

class NeuralEvent:
    """Represents a neural event."""
    def __init__(self, event_type, data=None):
        self.event_type = event_type
        self.data = data

class NeuralEventPublisher:
    """Manages subscribers and publishes neural events."""
    def __init__(self):
        self._subscribers = collections.defaultdict(list)

    def subscribe(self, event_type, subscriber):
        """Registers a subscriber for a specific event type."""
        if subscriber not in self._subscribers[event_type]:
            self._subscribers[event_type].append(subscriber)

    def unsubscribe(self, event_type, subscriber):
        """Unregisters a subscriber for a specific event type."""
        if subscriber in self._subscribers[event_type]:
            self._subscribers[event_type].remove(subscriber)

    def publish(self, event: NeuralEvent):
        """Publishes an event to all registered subscribers for that event type."""
        for subscriber in self._subscribers.get(event.event_type, []):
            try:
                subscriber.handle_event(event)
            except Exception as e:
                print(f"Error handling event {event.event_type} by {subscriber}: {e}")

# Example Subscriber Interface (for guidance, not strictly enforced by Python)
class NeuralEventSubscriber:
    """Interface for neural event subscribers."""
    def handle_event(self, event: NeuralEvent):
        """Method to be called when an event is published."""
        raise NotImplementedError("Subclasses must implement handle_event method")