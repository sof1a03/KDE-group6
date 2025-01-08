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
  lorem = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
  dummy_data = [
    ["Flowers for Algernon", "https://media.s-bol.com/DY9K0jOzlggk/wV6q0xg/550x836.jpg" ],
    ["Lord of the Rings", "https://m.media-amazon.com/images/I/913sMwNHsBL._SL1500_.jpg" ],
    ["The Perfume", "https://upload.wikimedia.org/wikipedia/en/f/f5/PerfumeSuskind.jpg" ],
    ["Fall or Dodge; in Hell", "https://m.media-amazon.com/images/I/91Z1I-2LQpL._AC_UF1000,1000_QL80_.jpg" ],
    ["Brave New World", "https://m.media-amazon.com/images/I/91D4YvdC0dL._AC_UF894,1000_QL80_.jpg" ],
    ["The Midnight Library", "https://media.s-bol.com/yyE0mpJQW856/o2jMxV3/547x840.jpg" ],
    ["Cryptonomicon", "https://m.media-amazon.com/images/I/811HmmUbx3L._AC_UF1000,1000_QL80_.jpg" ],
    ["Hersenschimmen", "https://media.s-bol.com/vZg12A3m36QM/739x1200.jpg" ],
    ["Tokyo Vice", "https://m.media-amazon.com/images/I/61TWK--pVVL._AC_UF894,1000_QL80_.jpg" ]
  ]

}
