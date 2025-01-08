import { Component } from '@angular/core';
import { BookCardComponent } from "../book-card/book-card.component";
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-book-view',
  imports: [BookCardComponent, CommonModule],
  templateUrl: './book-view.component.html',
  styleUrl: './book-view.component.css',
  standalone: true
})


export class BookViewComponent {
  dummy_data = [
    ["Flowers for Algernon", "https://media.s-bol.com/DY9K0jOzlggk/wV6q0xg/550x836.jpg" ],
    ["Flowers for Algernon", "https://media.s-bol.com/DY9K0jOzlggk/wV6q0xg/550x836.jpg" ],
    ["Flowers for Algernon", "https://media.s-bol.com/DY9K0jOzlggk/wV6q0xg/550x836.jpg" ],
    ["Flowers for Algernon", "https://media.s-bol.com/DY9K0jOzlggk/wV6q0xg/550x836.jpg" ],
    ["Flowers for Algernon", "https://media.s-bol.com/DY9K0jOzlggk/wV6q0xg/550x836.jpg" ],
    ["Flowers for Algernon", "https://media.s-bol.com/DY9K0jOzlggk/wV6q0xg/550x836.jpg" ],
    ["Flowers for Algernon", "https://media.s-bol.com/DY9K0jOzlggk/wV6q0xg/550x836.jpg" ],
    ["Flowers for Algernon", "https://media.s-bol.com/DY9K0jOzlggk/wV6q0xg/550x836.jpg" ],
    ["Flowers for Algernon", "https://media.s-bol.com/DY9K0jOzlggk/wV6q0xg/550x836.jpg" ]
  ]

}
